# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for video forecasting models."""

from concurrent.futures import thread
import functools as ft
import os
import time
from typing import Any, TypeVar

from absl import logging
import chex
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from connectomics.jax import checkpoint
from connectomics.jax import training
import flax
import flax.linen as nn
import jax
from jax import sharding
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from zapbench.video_forecasting import data_loading
from zapbench.video_forecasting import metrics as vid_metrics
import zapbench.models.util as model_util
import zapbench.video_forecasting.losses as cat_transforms

T = TypeVar('T', bound='metrics.Collection')


class TrainState(flax.struct.PyTreeNode):
  step: jnp.ndarray
  opt_state: optax.OptState
  params: flax.core.FrozenDict[str, Any]
  batch_stats: Any


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: chex.Array,
    item_shape: dict[str, tuple[int, ...]],
) -> tuple[nn.Module, optax.GradientTransformation, optax.Schedule, TrainState]:
  """Instantiates and initializes the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    item_shape: Shapes of the inputs fed into the model.

  Returns:
    The initialized TrainState with the optimizer.
  """
  logging.info('item_shape=%r', item_shape)
  model = model_util.model_from_config(config)
  if config.model_class == 'nunet.Nunet':
    num_dims = len(item_shape['input_frames']) - 2
    min_shape = (1, config.data_config.timesteps_input)
    num_scales = len(config.nunet_config.resample_factors) + 1
    if num_scales > 1:
      if config.nunet_config.num_maxvit_blocks > 0:
        patch_size = config.nunet_config.maxvit_patch_size
      else:
        patch_size = (1,) * num_dims
      assert len(patch_size) == num_dims
      total_factor = [
          np.prod([f[i] for f in config.nunet_config.resample_factors])
          for i in range(num_dims)
      ]
      min_shape += tuple(f * p for f, p in zip(total_factor, patch_size))
      min_shape += (1,)
    else:
      if config.nunet_config.num_maxvit_blocks > 0:
        patch_size = config.nunet_config.maxvit_patch_size
      else:
        patch_size = (1,) * num_dims
      assert len(patch_size) == num_dims
      min_shape += patch_size + (1,)
    cin_shape = (1,) + item_shape['input_stimulus']
    cout_shape = (1,) + item_shape['output_stimulus']
    t_shape = (1,) + item_shape['lead_time']
    inputs = (
        jnp.empty(min_shape),
        jnp.empty(cin_shape),
        jnp.empty(cout_shape),
        jnp.empty(t_shape),
    )
  elif config.model_class == 'nlinear.VideoNlinearGlobalUnivariate':
    inputs = (jnp.empty((1, config.data_config.timesteps_input, 1)),)
  else:
    inputs = (jnp.empty((1,) + item_shape['input_frames']),)
  variables = model.init(rng, *inputs, train=False)
  params = variables['params']
  parameter_overview.log_parameter_overview(params)
  optimizer, schedule = training.get_optimizer(config)
  opt_state = optimizer.init(params)
  n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
  logging.info('param count=%r', n_params)

  return (
      model,
      optimizer,
      schedule,
      TrainState(
          step=jnp.array(0, dtype=jnp.int32),
          opt_state=opt_state,
          batch_stats=variables.get('batch_stats', None),
          params=params,
      ),
  )


def inherit_annotations(cls: type[T]) -> type[T]:
  """Utility to inherit dataclass annotations of direct parent."""
  cls.__annotations__.update(super(cls, cls).__annotations__)
  return cls


@flax.struct.dataclass
class BaseMetrics(metrics.Collection):
  """Base metrics for video prediction."""

  learning_rate: metrics.LastValue.from_output('learning_rate')
  loss: metrics.Average.from_output('loss')
  loss_std: metrics.Std.from_output('loss')
  mse: metrics.Average.from_fun(vid_metrics.mse)
  mae: metrics.Average.from_fun(vid_metrics.mae)
  psnr: metrics.Average.from_fun(vid_metrics.psnr)


@flax.struct.dataclass
@inherit_annotations
class DetailedMetrics(BaseMetrics):
  """Detailed but expensive metrics for video prediction."""

  trace_mse: metrics.Average.from_fun(
      vid_metrics.make_trace_based_metric(vid_metrics.mse)
  )
  trace_mae: metrics.Average.from_fun(
      vid_metrics.make_trace_based_metric(vid_metrics.mae)
  )


def loss_and_predictions(
    criterion: str,
    predictions: chex.Array,
    targets: chex.Array,
    mask: chex.Array,
    trace_mask: chex.Array,
    trace_counts: chex.Array,
) -> tuple[chex.Array, chex.Array]:
  """Computes loss and transformed predictions for a given criterion.

  Feature shape should be 1 for mae/mse and number of bins for categorical case.

  Args:
    criterion: str denoting loss criterion
    predictions: (batch, t, x, y, z) + (feature_shape,)
    targets: (batch, t, x, y, z, 1)
    mask: binary of shape (1, 1, x, y, z, 1)
    trace_mask: trace mask
    trace_counts: trace counts

  Returns:
    loss: scalar
    predictions: (batch, t, x, y, (z,), 1)
  """
  # batch * t * non-zero elements
  num_terms_observed = predictions.shape[0] * predictions.shape[1] * mask.sum()
  num_terms = np.prod(predictions.shape[:-1])  # without feature shape
  loss_factor = num_terms / num_terms_observed
  if criterion in ['mse', 'mae']:
    delta = (predictions - targets) * mask
    if criterion == 'mae':
      loss = jnp.abs(delta).mean() * loss_factor
    else:  # 'mse'
      loss = jnp.square(delta).mean() * loss_factor
  elif criterion == 'trace_mae':
    loss_fn = vid_metrics.make_trace_based_metric(vid_metrics.mae)
    loss = loss_fn(predictions, targets, trace_mask, trace_counts).mean()
  elif 'hlgauss' in criterion:
    num_bins = predictions.shape[-1]
    hlg_transform = cat_transforms.GaussianHistogramLoss(
        num_bins=num_bins, sigma_ratio=0.75, min_value=-0.25, max_value=1.5
    )
    target_shape = targets.shape
    target_probs = jax.vmap(hlg_transform.transform_to_probs)(targets.flatten())
    target_probs = target_probs.reshape(target_shape[:-1] + (num_bins,))
    losses = optax.losses.softmax_cross_entropy(predictions, target_probs)
    if 'trace' in criterion:
      loss = vid_metrics.extract_traces(losses, trace_mask, trace_counts).mean()
    else:
      loss = (losses * mask.squeeze(axis=-1)).mean() * loss_factor
    prediction_shape = predictions.shape
    predictions = jax.vmap(hlg_transform.transform_from_probs)(
        nn.softmax(predictions.reshape(-1, num_bins))
    )
    predictions = predictions.reshape(prediction_shape[:-1] + (1,))
  else:
    raise ValueError(f'Unknown loss: {criterion}')
  return loss, predictions


def train_step(
    model: nn.Module,
    state: TrainState,
    batch: dict[str, chex.Array],
    config: ml_collections.ConfigDict,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    dropout_rng: jax.Array,
    trace_mask: data_loading.TraceMask,
    loss_mask: chex.Array,
    input_mask: chex.Array,
    frame_sharding: sharding.Sharding,
    stim_sharding: sharding.Sharding,
) -> tuple[TrainState, metrics.Collection]:
  """Performs a single training step.

  Args:
    model: Module to compute predictions.
    state: Current training state. Updated training state will be returned.
    batch: Training inputs for this step.
    config: Configuration for model.
    optimizer: optax optimizer.
    schedule: optax learning rate schedule.
    dropout_rng: RNG key for dropout.
    trace_mask: mask containing segmentations of traces
    loss_mask: binary mask on loss terms (0 indicates background)
    input_mask: multiplicative mask for the inputs.
    frame_sharding: how to shard video inputs and intermediate layers
    stim_sharding: how to shard the stimulus

  Returns:
    tuple of: updated state, dictionary with metrics
  """
  step = state.step + 1
  dropout_rng = jax.random.fold_in(dropout_rng, step)

  logging.log_first_n(
      logging.INFO, 'Inputs shape: %r', 1, batch['input_frames'].shape
  )
  logging.log_first_n(
      logging.INFO, 'Targets shape: %r', 1, batch['output_frames'].shape
  )

  def loss_fn(params):
    variables = {'params': params}
    if state.batch_stats is not None:
      variables['batch_stats'] = state.batch_stats

    # Main model loss.
    predictions, new_variables = model.apply(
        variables,
        batch['input_frames'] * input_mask,
        batch['input_stimulus'],
        batch['output_stimulus'],
        batch['lead_time'],
        x_sharding=frame_sharding,
        cond_sharding=stim_sharding,
        mutable=True,
        train=True,
        rngs={'dropout': dropout_rng},
    )

    logging.log_first_n(logging.INFO, 'Preds shape: %r', 1, predictions.shape)
    loss, predictions = loss_and_predictions(
        config.criterion,
        predictions,
        batch['output_frames'],
        loss_mask,
        trace_mask.segment_ids,
        trace_mask.counts,
    )

    return loss, (new_variables.get('batch_stats', None), predictions)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_batch_stats, predictions)), grad = grad_fn(state.params)

  updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
  params = optax.apply_updates(state.params, updates)

  new_state = state.replace(
      step=step, params=params, opt_state=opt_state, batch_stats=new_batch_stats
  )

  metrics_kwargs = dict(
      loss=loss,
      predictions=predictions,
      targets=batch['output_frames'],
      learning_rate=schedule(state.step - 1),
      video=True,
      has_channel=True,
      dynamic_range=config.data_config.dynamic_range,
      trace_mask=trace_mask.segment_ids,
      trace_counts=trace_mask.counts,
  )
  metrics_cls = DetailedMetrics if config.detailed_metrics else BaseMetrics
  metrics_update = metrics_cls.single_from_model_output(**metrics_kwargs)
  return new_state, metrics_update  # pytype: disable=bad-return-type


def eval_step(
    model: nn.Module,
    state: TrainState,
    batch: dict[str, chex.Array],
    config: ml_collections.ConfigDict,
    trace_mask: data_loading.TraceMask,
    loss_mask: chex.Array,
    input_mask: chex.Array,
    frame_sharding: sharding.Sharding,
    stim_sharding: sharding.Sharding,
) -> metrics.Collection:
  """Performs a single evaluation step.

  Args:
    model: Module to compute predictions.
    state: Current training state. Updated training state will be returned.
    batch: Training inputs for this step.
    config: Configuration for model.
    trace_mask: mask containing segmentations of traces
    loss_mask: mask used during training to filter out losses
    input_mask: multiplicative mask for the inputs.
    frame_sharding: how to shard video inputs and intermediate layers
    stim_sharding: how to shard the stimulus

  Returns:
    metrics
  """
  predictions = model.apply(
      {'params': state.params},
      batch['input_frames'] * input_mask,
      batch['input_stimulus'],
      batch['output_stimulus'],
      batch['lead_time'],
      x_sharding=frame_sharding,
      cond_sharding=stim_sharding,
      train=False,
      mutable=False,
  )
  loss, predictions = loss_and_predictions(
      config.criterion,
      predictions,
      batch['output_frames'],
      loss_mask,
      trace_mask.segment_ids,
      trace_mask.counts,
  )
  metrics_kwargs = dict(
      loss=loss,
      predictions=predictions,
      targets=batch['output_frames'],
      learning_rate=0.0,
      video=True,
      has_channel=True,
      dynamic_range=config.data_config.dynamic_range,
      trace_mask=trace_mask.segment_ids,
      trace_counts=trace_mask.counts,
  )
  metrics_cls = DetailedMetrics if config.detailed_metrics else BaseMetrics
  return metrics_cls.single_from_model_output(**metrics_kwargs)


def init_model(
    config: ml_collections.ConfigDict,
    model_rng: chex.Array,
    checkpoint_dir: str,
    item_shape: dict[str, tuple[int, ...]],
) -> tuple[
    nn.Module,
    TrainState,
    checkpoint.MixedMultihostCheckpoint,
    optax.GradientTransformation,
    optax.Schedule,
]:
  """Initializes the model.

  Args:
    config: configuration for model
    model_rng: rng for weight initialization
    checkpoint_dir: directory to store checkpoints in
    item_shape: shapes of the items contained in a batch

  Returns:
    tuple of: model, train state, checkpoint
  """
  model, optimizer, schedule, state = create_train_state(
      config, model_rng, item_shape
  )

  # Set up checkpointing of the model and the input pipeline.
  ckpt = checkpoint.MixedMultihostCheckpoint(checkpoint_dir, max_to_keep=100000)

  # If an initial checkpoint is provided and the checkpointing library does not
  # report a 'latest' checkpoint, then we are starting a new experiment.
  # Otherwise an existing experiment is being resumed (e.g. after the training
  # task being preempted) and the latest checkpoint should take precedence.
  latest = ckpt.get_latest_checkpoint_to_restore_from()
  if config.init_from_cpoint and latest is None:
    logging.info('Restoring model state from %s', config.init_from_cpoint)
    state = ckpt.restore(state, config.init_from_cpoint)
  else:
    logging.info('Restoring model state from %r', latest)
    state = ckpt.restore_or_initialize(state)

  return model, state, ckpt, optimizer, schedule


def get_shardings(
    config: ml_collections.ConfigDict,
) -> tuple[
    sharding.NamedSharding,
    sharding.NamedSharding,
    sharding.NamedSharding,
]:
  """Get global mesh and default replication and distribution shardings."""
  devices = np.array(jax.devices()).reshape(config.mesh_shape)
  mesh = sharding.Mesh(devices, config.mesh_names)
  replicate_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec())
  frame_sharding = sharding.NamedSharding(
      mesh, sharding.PartitionSpec(*config.mesh_names_batch)
  )
  stim_sharding = sharding.NamedSharding(
      mesh,
      sharding.PartitionSpec(*config.mesh_names_stimulus),
  )
  return replicate_sharding, frame_sharding, stim_sharding


def get_mask(
    mask_type: str,
    masks: data_loading.Masks,
    subsample: bool = False
) -> chex.Array:
  """Choose mask and expand to frame shape."""
  if mask_type == 'none':
    loss_mask = np.ones_like(masks.brain_mask)
  elif mask_type == 'brain':
    loss_mask = masks.brain_mask
  elif mask_type == 'trace':
    loss_mask = masks.trace_mask.segment_ids
  else:
    raise ValueError(f'Unknown mask: {mask_type}')

  if subsample:
    s = masks.out_to_in_scale_xyz
    logging.info('Downsampling mask by %r', s)
    loss_mask = loss_mask[::s[0], ::s[1], ::s[2]]

  # make mask binary and expand to have batch, time, and channel dimensions
  loss_mask = (loss_mask > 0).astype(bool)
  return np.expand_dims(loss_mask, axis=(0, 1, -1))


def make_tboard_compatible(
    perfs: dict[str, chex.Array], prefix: str = ''
) -> dict[str, chex.Array]:
  """Convert arrays to scalars using additional dictionary keys."""
  perfs_compat = dict()
  for k, v in perfs.items():
    if v.ndim == 0:
      perfs_compat[f'{prefix}/{k}'] = v
    elif v.ndim == 1:
      for t, vt in enumerate(v):
        perfs_compat[f'{prefix}_{k}/dt={t+1}'] = vt
    else:
      raise ValueError('Only scalars or vectors allowed for performances.')
  return perfs_compat


def sample_lead_time(config: ml_collections.ConfigDict) -> bool:
  if config.model_class == 'nunet.Nunet':
    return config.nunet_config.time_conditioning
  else:
    return False


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Main training loop."""
  tf.io.gfile.makedirs(workdir)
  rng = jax.random.PRNGKey(config.seed)
  logging.info('Workdir: %s', workdir)

  # Set up device mesh.
  replicate_sharding, frame_sharding, stim_sharding = get_shardings(config)

  loader_kwargs = dict(
      config=config,
      frame_sharding=frame_sharding,
      stim_sharding=stim_sharding,
      lead_time_sharding=replicate_sharding,
      sample_lead_time=sample_lead_time(config),
  )
  # Initialize training loaders.
  train_loader, masks = data_loading.get_dataset(
      num_steps=config.num_train_steps,
      split=config.train_split,
      return_masks=True,
      **loader_kwargs,
  )
  trace_mask = masks.trace_mask
  num_eval_steps = (
      config.num_train_steps // config.eval_every_steps + 1
  ) * config.num_eval_steps
  eval_iter = iter(
      data_loading.get_dataset(
          num_steps=num_eval_steps, split=config.val_split, **loader_kwargs
      )
  )

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  cpoint_dir = os.path.join(workdir, 'checkpoints')
  model, state, ckpt, optimizer, schedule = init_model(
      config,
      model_rng,
      cpoint_dir,
      train_loader.tensor_source.item_shape,
  )
  train_loader.set_initial_batch(int(state.step))
  _, dropout_rng = jax.random.split(rng)
  initial_step = int(state.step) + 1

  def replicate(arr):
    addressable_devices = replicate_sharding.addressable_devices
    arrs = [jax.device_put(arr, d) for d in addressable_devices]
    return jax.make_array_from_single_device_arrays(
        arr.shape, replicate_sharding, arrs
    )

  # TODO(aleximmer): in case we end up using the local univariate model, use
  # flat sharding on the first dimension of parameter shapes.
  state = jax.tree.map(replicate, state)
  trace_mask = jax.tree.map(replicate, trace_mask)
  loss_mask = jax.tree.map(replicate, get_mask(config.loss_mask, masks))
  input_mask = jax.tree.map(replicate,
                            get_mask(config.input_mask, masks, subsample=True))
  dropout_rng = jax.tree.map(replicate, dropout_rng)

  def train_fn(
      state: TrainState,
      batch: dict[str, chex.Array],
      dropout_rng: chex.Array,
  ):
    return train_step(
        model,
        state,
        batch,
        config,
        optimizer,
        schedule,
        dropout_rng,
        trace_mask,
        loss_mask,
        input_mask,
        frame_sharding,
        stim_sharding,
    )

  def eval_fn(
      state: TrainState,
      batch: dict[str, chex.Array],
  ):
    return eval_step(
        model,
        state,
        batch,
        config,
        trace_mask,
        loss_mask,
        input_mask,
        frame_sharding,
        stim_sharding,
    )

  batch_sharding = dict(
      input_frames=frame_sharding,
      output_frames=frame_sharding,
      input_stimulus=stim_sharding,
      output_stimulus=stim_sharding,
      lead_time=replicate_sharding,
  )

  in_shardings = (
      replicate_sharding,  # state (params)
      batch_sharding,  # inputs
      replicate_sharding,  # dropout rngs
  )
  out_shardings = (
      replicate_sharding,  # params
      replicate_sharding,  # metrics
  )

  in_shardings_eval = (replicate_sharding, batch_sharding)
  out_shardings_eval = replicate_sharding

  def to_global_array(batch):
    return {k: v.to_global() for k, v in batch.items()}  # pytype: disable=attribute-error

  def free_and_delete(batch):
    for k in batch:
      batch[k].delete()
    del batch

  train_jit = jax.jit(train_fn, in_shardings, out_shardings)
  eval_jit = jax.jit(eval_fn, in_shardings_eval, out_shardings_eval)

  # Initialize summary writer.
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if initial_step == 1:
    writer.write_hparams({
        k: v
        for k, v in config.items()
        if isinstance(v, (bool, float, int, str))
    })

  model_util.save_config(config, os.path.join(workdir, 'config.json'))
  logging.info('Saved model config.')

  logging.info('Starting training loop at step %d.', initial_step)
  hooks = []
  model_step = 0  # counts train and eval steps together
  report_progress = training.ReportProgress(
      config.global_batch_size,
      num_train_steps=config.num_train_steps + num_eval_steps,
      writer=writer,
  )
  if jax.process_index() == 0:
    hooks.append(report_progress)

  checkpoint_thread = thread.ThreadPoolExecutor(1, 'checkpoint')
  metrics_cls = DetailedMetrics if config.detailed_metrics else BaseMetrics
  train_metrics = jax.tree.map(replicate, metrics_cls.empty())

  step = None
  shutdown_request = False

  with metric_writers.ensure_flushes(writer):
    steps = range(initial_step, config.num_train_steps + 1)
    for step, batch in zip(steps, train_loader):
      is_last_step = step == config.num_train_steps
      batch = to_global_array(batch)

      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        with report_progress.timed('train'):
          logging.log_first_n(
              logging.INFO,
              'Input %r. Output %r. Input stimulus: %r. Output stimulus: %r',
              1,
              batch['input_frames'].shape,
              batch['output_frames'].shape,
              batch['input_stimulus'].shape,
              batch['output_stimulus'].shape,
          )
          logging.log_first_n(
              logging.INFO, 'lead_time: %r', 10, batch['lead_time']
          )

          state, metrics_update = train_jit(state, batch, dropout_rng)

        with jax.spmd_mode('allow_all'):
          train_metrics = train_metrics.merge(metrics_update)

      model_step += 1
      for h in hooks:
        h(model_step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        with jax.spmd_mode('allow_all'):
          scalars = make_tboard_compatible(
              train_metrics.compute(), prefix='train'
          )
        writer.write_scalars(step, scalars)
        train_metrics = jax.tree.map(replicate, metrics_cls.empty())

      if step % config.eval_every_steps == 0 or is_last_step:
        # Free memory of batch arrays manually between train and eval
        free_and_delete(batch)

        with report_progress.timed('eval'):
          eval_metrics = jax.tree.map(replicate, metrics_cls.empty())
          for val_step in range(config.num_eval_steps):
            with jax.profiler.StepTraceAnnotation('eval', step_num=val_step):
              batch = to_global_array(next(eval_iter))
              logging.log_first_n(
                  logging.INFO,
                  'Input %r. Output %r. Input stim: %r. Output stim: %r',
                  1,
                  batch['input_frames'].shape,
                  batch['output_frames'].shape,
                  batch['input_stimulus'].shape,
                  batch['output_stimulus'].shape,
              )
              if config.eval_with_train_step:
                _, metrics_update = train_jit(state, batch, dropout_rng)
              else:
                metrics_update = eval_jit(state, batch)
              with jax.spmd_mode('allow_all'):
                eval_metrics = eval_metrics.merge(metrics_update)

            model_step += 1
            for h in hooks:
              h(model_step)

          with jax.spmd_mode('allow_all'):
            scalars = make_tboard_compatible(
                eval_metrics.compute(), prefix='eval'
            )
          writer.write_scalars(step, scalars)

        free_and_delete(batch)

      shutdown_request = multihost_utils.reached_preemption_sync_point(step)
      if (
          step % config.checkpoint_every_steps == 0
          or is_last_step
          or shutdown_request
      ):
        logging.info('Saving checkpoint at step %d.', step)

        def _save_cpoint(ckpt_state):
          with report_progress.timed('checkpoint'):
            ckpt.save(ckpt_state)

        with report_progress.timed('checkpoint_submit'):
          # move to host memory to reduce GPU RAM usage.
          ckpt_state = jax.tree.map(np.array, state)
          checkpoint_thread.submit(ft.partial(_save_cpoint, ckpt_state))

        logging.info('Checkpoint from step %d saved.', step)

      if shutdown_request:
        logging.warn('Interrupting training loop due to shutdown request.')
        logging.flush()
        break

  checkpoint_thread.shutdown()
  logging.info('Finished training at step %d.', step)

  if shutdown_request:
    # Allow time for other workers to finish checkpoint saving. Soon after
    # the first worker is terminated, it will be detected that the clique
    # is no longer complete, which will cause an immediate restart of the
    # current process via std::quick_exit(42).
    time.sleep(60)
