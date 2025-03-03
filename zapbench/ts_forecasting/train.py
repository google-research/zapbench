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

"""Methods for training."""

from collections.abc import Mapping, Sequence
import contextlib
import functools
import itertools as it
import os
from typing import Any

from absl import logging
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import parameter_overview
from clu import periodic_actions
from clu import platform  # pylint: disable=unused-import
from connectomics.jax import checkpoint
from connectomics.jax import training
import connectomics.jax.metrics as metrics_lib
from etils import epath
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import zapbench.models.util as model_util
from zapbench.ts_forecasting import heads
from zapbench.ts_forecasting import input_pipeline
from zapbench.ts_forecasting import util


@flax.struct.dataclass
class TrainState:
  """State of the model and the training.

  This includes parameters, statistics and optimizer.
  """

  step: int
  params: Any
  opt_state: optax.OptState
  batch_stats: Any
  dropout_key: jax.Array


def merge_batch_stats(replicated_state: TrainState) -> TrainState:
  """Merge model batch stats."""
  if jax.tree_util.tree_leaves(replicated_state.batch_stats):
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return replicated_state.replace(
        batch_stats=cross_replica_mean(replicated_state.batch_stats)
    )
  else:
    return replicated_state


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: jax.Array,
    input_shapes: Sequence[Sequence[int]],
) -> tuple[nn.Module, optax.GradientTransformation, optax.Schedule, TrainState]:
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shapes: Input shapes.

  Returns:
    The model, optimizer, and initial train state.
  """
  model = model_util.model_from_config(config)
  init_rng, dropout_rng = jax.random.split(rng, num=2)
  variables = model.init(
      init_rng, *[jnp.ones(s) for s in input_shapes], train=False
  )
  params = variables['params']
  batch_stats = variables.get('batch_stats', None)
  parameter_overview.log_parameter_overview(params)
  optimizer, schedule = training.get_optimizer(config)
  opt_state = optimizer.init(params)
  # pylint: disable=unexpected-keyword-arg
  return (
      model,
      optimizer,
      schedule,
      TrainState(
          step=0,
          params=params,
          opt_state=opt_state,
          batch_stats=batch_stats,
          dropout_key=dropout_rng,
      ),
  )
  # pylint: enable=unexpected-keyword-arg


def train_step(
    model: nn.Module,
    head: heads.Head,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    train_state: TrainState,
    batch: Mapping[str, jax.Array],
    covariates: Sequence[str],
) -> tuple[TrainState, clu_metrics.Collection]:
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input arrays
      and a boolean argument indicating whether to use training or inference
      mode.
    head: Model head.
    optimizer: Optax optimizer.
    schedule: Optax learning rate schedule.
    train_state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    covariates: Covariates to pass to the model.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.log_first_n(logging.INFO, 'train_step(batch=%s)', 1, batch)

  train_dropout_key = jax.random.fold_in(
      key=train_state.dropout_key, data=train_state.step
  )

  def loss_fn(params):
    variables = {'params': params}
    mutable = []
    if train_state.batch_stats is not None:
      variables['batch_stats'] = train_state.batch_stats
      mutable.append('batch_stats')
    predictions, new_variables = model.apply(
        variables,
        *[batch[k] for k in it.chain(('timeseries_input',), covariates)],
        rngs={'dropout': train_dropout_key},
        mutable=mutable,
        train=True,
    )
    loss = head.compute_loss(
        predictions=predictions, targets=batch['timeseries_output']
    )
    return loss, (new_variables.get('batch_stats', None), predictions)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_batch_stats, predictions)), grad = grad_fn(train_state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name='batch')
  updates, new_opt_state = optimizer.update(
      grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  new_state = train_state.replace(  # pytype: disable=attribute-error
      step=train_state.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      batch_stats=new_batch_stats,
  )

  metrics_update = head.metrics['train'].gather_from_model_output(
      loss=loss,
      predictions=predictions,
      targets=batch['timeseries_output'],
      learning_rate=schedule(train_state.step),
  )
  return new_state, metrics_update


@functools.partial(jax.jit, static_argnums=(0, 3, 5))
def pred_step(
    model: nn.Module,
    train_state: TrainState,
    batch: Mapping[str, jax.Array],
    covariates: Sequence[str],
    initial_carry: jax.Array | None = None,
    return_carry: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Perform a single prediction step.

  Args:
    model: Flax module for the model. The apply method must take input arrays
      and a boolean argument indicating whether to use training or inference
      mode.
    train_state: Replicate model state.
    batch: Inputs.
    covariates: Covariates to pass to the model.
    initial_carry: If not None, passed to model as `initial_carry` keyword
      argument which can e.g. be passed on further to `nn.RNN`.
    return_carry: If True, passed to model as `return_carry` keyword argument
      which can e.g. be passed on further to `nn.RNN`. In this case, the model
      should return a pair of arrays rather than a single array.

  Returns:
    A single array containing the model output if `return_carry` is `False`,
    otherwise, a pair of arrays containing carry and model outputs.
  """
  logging.log_first_n(logging.INFO, 'pred_step(batch=%s)', 1, batch)
  variables = {
      'params': train_state.params,
  }
  if train_state.batch_stats is not None:
    variables['batch_stats'] = train_state.batch_stats
  model_kwargs = {
      'mutable': False,
      'train': False,
  }
  if initial_carry is not None:
    model_kwargs['initial_carry'] = initial_carry
  if return_carry:
    model_kwargs['return_carry'] = return_carry
  return model.apply(
      variables,
      *[batch[k] for k in it.chain(('timeseries_input',), covariates)],
      **model_kwargs,
  )


def val_step(
    model: nn.Module,
    head: heads.Head,
    train_state: TrainState,
    batch: Mapping[str, jax.Array],
    covariates: Sequence[str],
) -> clu_metrics.Collection:
  """Compute the metrics for the given model without training.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input arrays
      and a boolean argument indicating whether to use training or inference
      mode.
    head: Model head.
    train_state: Replicate model state.
    batch: Inputs.
    covariates: Covariates to pass to the model.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.log_first_n(logging.INFO, 'val_step(batch=%s)', 1, batch)
  predictions = pred_step(model, train_state, batch, covariates)
  return head.metrics['val'].gather_from_model_output(
      predictions=predictions,
      targets=batch['timeseries_output'],
      loss=head.compute_loss(
          predictions=predictions, targets=batch['timeseries_output']
      ),
  )


def _log_nvidia_smi(*unused_args, **unused_kwargs):
  logging.info(util.try_nvidia_smi())


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: epath.PathLike
) -> Mapping[str, Any]:
  """Runs a training and validation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.

  Returns:
    Checkpoint dictionary.
  """
  workdir = epath.Path(workdir)
  workdir.mkdir(parents=True, exist_ok=True)

  # Seeding.
  rng = training.get_rng(config.seed)
  logging.info('Using random seed %s.', rng)

  # Model seed.
  rng, model_rng = jax.random.split(rng)

  # Build input pipeline.
  rng, data_seed = jax.random.split(rng)
  data_seed = int(
      jax.random.randint(data_seed, [], minval=0, maxval=np.iinfo(np.int32).max)
  )
  train_loader, num_train_records, val_loader, _ = (
      input_pipeline.create_datasets(config, data_seed)
  )

  # Static covariates.
  covariates_static = input_pipeline.get_static_covariates(config)

  # Calculate number of training steps.
  num_train_steps = input_pipeline.get_num_train_steps(
      num_train_records, config
  )
  steps_per_epoch = num_train_steps // config.num_epochs
  logging.info(
      'num_train_steps=%d, steps_per_epoch=%d', num_train_steps, steps_per_epoch
  )

  # Initialize model.
  model, optimizer, schedule, train_state = create_train_state(
      config,
      model_rng,
      input_shapes=(config.series_shape,) + config.covariates_shapes,
  )

  # Initialize head.
  head = heads.create_head(config)

  # Early stopping.
  early_stop = util.EarlyStoppingWithStep(
      min_delta=config.early_stopping_min_delta,
      patience=config.early_stopping_patience,
  )

  # Max runtime stopping.
  max_runtime_stop = util.MaxRuntimeStopping(
      max_runtime=config.max_runtime
  ).set_reference_timestamp()

  # Tracker for step at which validation loss was observered.
  track_best_val_loss_step = util.TrackBestStep()

  # Set up checkpointing.
  checkpoint_manager = checkpoint.get_checkpoint_manager(
      workdir,
      item_names=(
          'early_stop',
          'train_state',
          'max_runtime_stop',
          'train_iter',  # pygrain
          'track_best_val_loss_step',
      ),
  )

  # Retrieve data from previous checkpoints if possible.
  train_iter = iter(train_loader)
  checkpointed_state = dict(
      early_stop=early_stop,
      train_state=train_state,
      train_iter=train_iter,
      max_runtime_stop=max_runtime_stop,
      track_best_val_loss_step=track_best_val_loss_step,
  )
  if checkpoint_manager.latest_step() is not None:
    checkpointed_state = checkpoint.restore_checkpoint(
        checkpoint_manager,
        state=checkpointed_state,
        step=checkpoint_manager.latest_step(),
        pygrain_checkpointers=('train_iter',),
    )
  early_stop = checkpointed_state['early_stop']
  train_state = checkpointed_state['train_state']
  train_iter = checkpointed_state['train_iter']
  max_runtime_stop = checkpointed_state['max_runtime_stop']
  track_best_val_loss_step = checkpointed_state['track_best_val_loss_step']

  # Distribute training.
  train_state = flax_utils.replicate(train_state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          head=head,
          optimizer=optimizer,
          schedule=schedule,
          covariates=config.covariates,
      ),
      axis_name='batch',
  )

  # Distribute validation.
  p_val_step = jax.pmap(
      functools.partial(
          val_step,
          model=model,
          head=head,
          covariates=config.covariates,
      ),
      axis_name='batch',
  )

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    # Add hooks.
    hooks += [
        report_progress,
    ]
    if config.periodic_profiling:
      hooks += [
          periodic_actions.Profile(num_profile_steps=5, logdir=workdir),
      ]
    if config.periodic_nvidia_smi:
      hooks += [
          periodic_actions.PeriodicCallback(
              every_steps=config.periodic_nvidia_smi,
              callback_fn=_log_nvidia_smi,
          )
      ]

    model_util.save_config(config, os.path.join(workdir, 'config.json'))
    logging.info('Saved model config.')

  train_metrics = None
  # Unreplicating from TPU is costly, so we only do it once at the start.
  initial_step, step = int(flax.jax_utils.unreplicate(train_state.step)), None
  with metric_writers.ensure_flushes(writer):
    # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
    for step in range(initial_step + 1, num_train_steps + 1):
      is_last_step, last_step_message = False, ''

      # num_train_steps stopping.
      if step == num_train_steps:
        is_last_step = True
        last_step_message = (
            'Met total number of training steps stopping criterion at step %s.'
            % step
        )
        logging.info(last_step_message)

      if step == 1:
        writer.write_hparams(dict(config))

      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        batch = next(train_iter)
        if (
            'covariates_static' in config.covariates
            and covariates_static is not None
        ):
          batch['covariates_static'] = covariates_static.repeat(
              jax.local_device_count(), axis=0)  # pytype: disable=unsupported-operands
        batch = training.reshape_batch_local_devices(batch)
        train_state, metrics_update = p_train_step(
            train_state=train_state, batch=batch
        )
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None
            else train_metrics.merge(metric_update)
        )

      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)

      for h in hooks:
        h(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        train_metrics_cpu = jax.tree.map(np.array, train_metrics.compute())
        writer.write_scalars(
            step, metrics_lib.make_dict_of_scalars(train_metrics_cpu)
        )
        train_metrics = None
        if config.early_stopping_metric in train_metrics_cpu:
          _, early_stop = early_stop.update(
              train_metrics_cpu[config.early_stopping_metric], step
          )

      # Early stopping.
      if config.early_stopping and early_stop.should_stop:
        is_last_step = True
        last_step_message = 'Met early stopping criterion at step %d.' % step
        logging.info(last_step_message)

      # Max runtime stopping.
      max_runtime_stop = max_runtime_stop.update()
      if config.max_runtime_stopping and max_runtime_stop.should_stop:
        is_last_step = True
        last_step_message = (
            'Met max runtime stopping criterion at step %d.' % step
        )
        logging.info(last_step_message)

      if is_last_step:
        platform.work_unit().set_notes(
            last_step_message
            + ' Final round of validation.'
        )

      # Validation.
      if (step % config.val_every_steps == 0 or is_last_step) and (
          config.num_val_steps != 0
      ):
        logging.info('Starting validation at step %d.', step)
        with (
            report_progress.timed('val')
            if not last_step_message
            else contextlib.nullcontext()
        ):
          val_metrics = None
          with training.StepTraceContextHelper('val', 0) as trace_context:
            # Use `iter` to reset the val_loader before each validation.
            for s, batch in enumerate(iter(val_loader)):
              if (
                  'covariates_static' in config.covariates
                  and covariates_static is not None
              ):
                batch['covariates_static'] = covariates_static.repeat(
                    jax.local_device_count(), axis=0)  # pytype: disable=unsupported-operands
              batch = training.reshape_batch_local_devices(batch)
              metrics_update = flax_utils.unreplicate(
                  p_val_step(
                      train_state=merge_batch_stats(train_state), batch=batch
                  )
              )
              val_metrics = (
                  metrics_update
                  if val_metrics is None
                  else val_metrics.merge(metrics_update)
              )
              if config.num_val_steps > 0 and s + 1 == config.num_val_steps:
                break
              trace_context.next_step()
          if val_metrics is None:
            raise ValueError(f'Val dataset {val_loader} was empty.')

        val_metrics_cpu = jax.tree.map(np.array, val_metrics.compute())
        writer.write_scalars(
            step, metrics_lib.make_dict_of_scalars(val_metrics_cpu)
        )
        track_best_val_loss_step = track_best_val_loss_step.update(
            val_metrics_cpu[config.early_stopping_metric], step
        )
        if config.early_stopping_metric in val_metrics_cpu:
          _, early_stop = early_stop.update(
              val_metrics_cpu[config.early_stopping_metric], step
          )
        ran_validation = True
      else:
        ran_validation = False

      # Checkpointing.
      if (
          step % config.checkpoint_every_steps == 0
          or is_last_step
          or ran_validation
      ):
        with (
            report_progress.timed('checkpoint')
            if not last_step_message
            else contextlib.nullcontext()
        ):
          train_state = merge_batch_stats(train_state)
          checkpoint.save_checkpoint(
              checkpoint_manager,
              state=dict(
                  early_stop=early_stop,
                  train_state=jax.tree.map(
                      np.array, flax_utils.unreplicate(train_state)
                  ),
                  train_iter=train_iter,
                  max_runtime_stop=max_runtime_stop,
                  track_best_val_loss_step=track_best_val_loss_step,
              ),
              step=step,
              pygrain_checkpointers=('train_iter',),
              wait_until_finished=True)

      if is_last_step:
        break

    if step:
      logging.info('Training/validation finished at step %d.', step)

  return checkpointed_state
