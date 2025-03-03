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

"""Video prediction into a tensorstore volume for inspection."""

import collections
import os

from absl import logging
import chex
from clu import metric_writers
from clu import metrics
from connectomics.common import ts_utils
from connectomics.jax import training
import flax
import flax.linen as nn
import jax
from jax import sharding
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_collections
import ml_collections.config_flags  # pylint:disable=unused-import
import numpy as np
import tensorstore as ts
from zapbench import constants
import zapbench.models.util as model_util
from zapbench.video_forecasting import data_loading
from zapbench.video_forecasting import metrics as vid_metrics
from zapbench.video_forecasting import train


@flax.struct.dataclass
class Metrics(metrics.Collection):
  """Evaluation metrics for video prediction."""

  step_mse: vid_metrics.PerStepAverage.from_fun(
      vid_metrics.make_per_step_metric(vid_metrics.mse)
  )
  step_mae: vid_metrics.PerStepAverage.from_fun(
      vid_metrics.make_per_step_metric(vid_metrics.mae)
  )
  step_psnr: vid_metrics.PerStepAverage.from_fun(
      vid_metrics.make_per_step_metric(vid_metrics.psnr)
  )
  trace_step_mse: vid_metrics.PerStepAverage.from_fun(
      vid_metrics.make_per_step_metric(
          vid_metrics.make_trace_based_metric_with_extracted_traces(
              vid_metrics.mse
          )
      )
  )
  trace_step_mae: vid_metrics.PerStepAverage.from_fun(
      vid_metrics.make_per_step_metric(
          vid_metrics.make_trace_based_metric_with_extracted_traces(
              vid_metrics.mae
          )
      )
  )


class ForecastWriter:
  """Base class for writing forecasts."""

  def _write(
      self, sample_ix: int, time_slice: slice, type_ix: int, value: chex.Array
  ) -> None:
    raise NotImplementedError()

  def write_true(
      self, sample_ix: int, time_slice: slice, value: chex.Array
  ) -> None:
    # write ground truth at index 0
    return self._write(sample_ix, time_slice, 0, value)

  def write_pred(
      self, sample_ix: int, time_slice: slice, value: chex.Array
  ) -> None:
    # write prediction at index 1
    return self._write(sample_ix, time_slice, 1, value)

  def flush(self):
    raise NotImplementedError()


class TsForecastWriter(ForecastWriter):
  """Writes forecasts to tensorstore."""

  def __init__(self, store: ts.TensorStore, max_queue: int):
    self.store = store
    self.queue = collections.deque()
    self.max_queue = max_queue

  def _write(
      self, sample_ix: int, time_slice: slice, type_ix: int, value: chex.Array
  ) -> None:
    if len(self.queue) >= self.max_queue:
      self.queue.popleft().result()
    write_future = self.store[sample_ix, time_slice, ..., type_ix].write(value)
    self.queue.append(write_future)

  def flush(self):
    while self.queue:
      self.queue.popleft().result()


class BlankForecastWriter(ForecastWriter):
  """Does not write anything."""

  def _write(
      self, sample_ix: int, time_slice: slice, type_ix: int, value: chex.Array
  ) -> None:
    pass

  def flush(self):
    pass


def predict(
    model: nn.Module,
    state: train.TrainState,
    sample: dict[str, chex.Array],
    config: ml_collections.ConfigDict,
    trace_mask: data_loading.TraceMask,
    loss_mask: chex.Array,
    input_mask: chex.Array,
    frame_sharding: sharding.Sharding,
    stim_sharding: sharding.Sharding,
    mask_video: bool,
) -> tuple[chex.Array, chex.Array, chex.Array, Metrics, train.TrainState]:
  """Performs a prediction step with evaluation.

  Args:
    model: Module to compute predictions.
    state: Current training state. Updated training state will be returned.
    sample: Training input for this step.
    config: Configuration for model.
    trace_mask: mask containing segmentations of traces
    loss_mask: mask used during training to filter out losse
    input_mask: multiplicative mask for the input.
    frame_sharding: how to shard video inputs and intermediate layers
    stim_sharding: how to shard the stimulus
    mask_video: whether to mask the video prediction

  Returns:
    tuple of: prediction, trace_prediction, trace_target, metrics
  """
  predictions = model.apply(
      {'params': state.params},
      sample['input_frames'] * input_mask,
      sample['input_stimulus'],
      sample['output_stimulus'],
      sample['lead_time'],
      x_sharding=frame_sharding,
      cond_sharding=stim_sharding,
      train=False,
      mutable=False,
  )
  logging.log_first_n(logging.INFO, 'Preds shape: %r', 1, predictions.shape)
  logging.log_first_n(
      logging.INFO, 'Targets shape: %r', 1, sample['output_frames'].shape
  )
  loss, predictions = train.loss_and_predictions(
      config.criterion,
      predictions,
      sample['output_frames'],
      loss_mask,
      trace_mask.segment_ids,
      trace_mask.counts,
  )
  if mask_video:
    predictions = predictions * loss_mask

  trace_predictions = vid_metrics.extract_traces(
      predictions,
      trace_mask.segment_ids[..., jnp.newaxis],
      trace_mask.counts,
  )
  trace_targets = vid_metrics.extract_traces(
      sample['output_frames'],
      trace_mask.segment_ids[..., jnp.newaxis],
      trace_mask.counts,
  )

  metrics_update = Metrics.single_from_model_output(
      loss=loss,
      predictions=predictions,
      targets=sample['output_frames'],
      video=True,
      dynamic_range=config.data_config.dynamic_range,
      trace_targets=trace_targets,
      trace_predictions=trace_predictions,
      nan_to_zero=True,
  )

  return predictions, trace_predictions, trace_targets, metrics_update, state


def predict_bw(
    model: nn.Module,
    state: train.TrainState,
    sample: dict[str, chex.Array],
    config: ml_collections.ConfigDict,
    trace_mask: data_loading.TraceMask,
    loss_mask: chex.Array,
    input_mask: chex.Array,
    frame_sharding: sharding.Sharding,
    stim_sharding: sharding.Sharding,
    mask_video: bool,
) -> tuple[chex.Array, chex.Array, chex.Array, Metrics, nn.Module]:
  """Performs prediction step with forced backward pass.

  Args:
    model: Module to compute predictions.
    state: Current training state. Updated training state will be returned.
    sample: Training input for this step.
    config: Configuration for model.
    trace_mask: mask containing segmentations of traces
    loss_mask: mask used during training to filter out losses
    input_mask: multiplicative mask for the input.
    frame_sharding: how to shard video inputs and intermediate layers
    stim_sharding: how to shard the stimulus
    mask_video: whether to mask the video prediction

  Returns:
    tuple of: prediction, trace_prediction, trace_target, metrics
  """

  def loss_fn(params):
    predictions = model.apply(
        {'params': params},
        sample['input_frames'] * input_mask,
        sample['input_stimulus'],
        sample['output_stimulus'],
        sample['lead_time'],
        x_sharding=frame_sharding,
        cond_sharding=stim_sharding,
        train=False,
        mutable=False,
    )
    logging.log_first_n(logging.INFO, 'Preds shape: %r', 1, predictions.shape)
    logging.log_first_n(
        logging.INFO, 'Targets shape: %r', 1, sample['output_frames'].shape
    )
    loss, predictions = train.loss_and_predictions(
        config.criterion,
        predictions,
        sample['output_frames'],
        loss_mask,
        trace_mask.segment_ids,
        trace_mask.counts,
    )
    if mask_video:
      predictions = predictions * loss_mask

    return loss, predictions

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, predictions), grad = grad_fn(state.params)

  trace_predictions = vid_metrics.extract_traces(
      predictions,
      trace_mask.segment_ids[..., jnp.newaxis],
      trace_mask.counts,
  )
  trace_targets = vid_metrics.extract_traces(
      sample['output_frames'],
      trace_mask.segment_ids[..., jnp.newaxis],
      trace_mask.counts,
  )

  metrics_update = Metrics.single_from_model_output(
      loss=loss,
      predictions=predictions,
      targets=sample['output_frames'],
      video=True,
      dynamic_range=config.data_config.dynamic_range,
      trace_targets=trace_targets,
      trace_predictions=trace_predictions,
      nan_to_zero=True,
  )

  return predictions, trace_predictions, trace_targets, metrics_update, grad


def write_metrics(
    writer: metric_writers.MetricWriter,
    temporal_metrics: dict[int, Metrics],
    num_pred_steps: int,
    timesteps_output: int,
):
  """Write out scalar performances for different lead times."""
  time_step = 1
  for i in range(num_pred_steps):
    scalars = temporal_metrics[i].compute()
    for t in range(timesteps_output):
      scalars_t = {k: v[t] for k, v in scalars.items()}
      writer.write_scalars(step=time_step, scalars=scalars_t)
      time_step += 1


def metrics_to_dict(
    temporal_metrics: dict[int, Metrics],
    num_pred_steps: int,
    timesteps_output: int,
):
  """Convert scalar performances for different lead times to dict."""
  metrics_dict = {}
  time_step = 1
  for i in range(num_pred_steps):
    scalars = temporal_metrics[i].compute()
    for _ in range(timesteps_output):
      for k, v in scalars.items():
        assert len(v) == 1
        metrics_dict[f'{k}/{time_step}'] = float(v[0])
      time_step += 1
  return metrics_dict


def combine_input_and_prediction(
    condition: chex.Array,
    prediction: chex.Array,
    config: ml_collections.ConfigDict,
) -> chex.Array:
  # combine along temporal axis and crop t
  t_in = config.data_config.timesteps_input
  with jax.spmd_mode('allow_all'):
    return jnp.concatenate([condition, prediction], axis=1)[:, -t_in:]


def infer(config: ml_collections.ConfigDict, infer_workdir: str):
  """Infer on the evaluation set and write forecast into tensorstore."""
  # Get experiment workdir and config.
  exp_workdir = (
      config.exp_workdir if config.exp_workdir else infer_workdir
  )
  exp_config = model_util.load_config(
      os.path.join(exp_workdir, 'config.json')
  )
  logging.info('Experiment workdir %r', exp_workdir)
  logging.info('Experiment config %r', exp_config)

  assert config.mesh_shape[0] == 1, 'Parallel batch inference unsupported.'
  replicate_sharding, frame_sharding, stim_sharding = train.get_shardings(
      config
  )

  sample_sharding = dict(
      input_frames=frame_sharding,
      output_frames=frame_sharding,
      input_stimulus=stim_sharding,
      output_stimulus=stim_sharding,
      lead_time=replicate_sharding,
  )

  prediction_window_length = constants.PREDICTION_WINDOW_LENGTH
  timesteps_output = exp_config.data_config.timesteps_output
  assert prediction_window_length % timesteps_output == 0
  num_pred_steps = prediction_window_length // timesteps_output
  sample_lead_time = train.sample_lead_time(exp_config)
  exp_config.data_config.timesteps_output = prediction_window_length
  exp_config.data_config.conditions = (config.condition,)
  exp_config.global_batch_size = 1
  exp_config.data_config.num_threads = 16  # to avoid OOMs
  loader, masks = data_loading.get_dataset(
      config=exp_config,
      num_steps=1,
      frame_sharding=frame_sharding,
      stim_sharding=stim_sharding,
      lead_time_sharding=replicate_sharding,
      split=config.split,
      contiguous_segments=False,  # use nan_to_zero for missed cells
      return_masks=True,
      sample_lead_time=False,  # manually set lead times for inference
      shuffle=False,
  )
  trace_mask = masks.trace_mask
  data_source = loader.tensor_source
  rng = jax.random.PRNGKey(exp_config.seed + jax.process_index())

  video_ts_path = (
      f'{config.base_path}{config.xm_id}/{config.work_unit}/{config.cpoint_id}'
      + f'/frames/{config.condition}/{config.split}'
  )

  if jax.process_index() == 0 and config.write_video_forecast:
    # create forecast tensorstore with [samples, x, y, z, t, true + pred]
    volume = ts.open(
        exp_config.data_config.tensorstore_input_config.to_dict()
    )
    metadata = volume.result().spec().to_json()['metadata']
    attributes = metadata['attributes']
    attributes['dimension_units'] = (
        ['0.9141s'] + attributes['dimension_units'] + ['']
    )
    metadata['chunk_grid']['configuration']['chunk_shape'] = [
        1,
        512,
        512,
        1,
        1,
        2,
    ]  # samples, x, y, z, lead time, true + pred
    metadata['codecs'][0]['configuration']['order'] = list(range(6))
    metadata['dimension_names'] = ['b'] + metadata['dimension_names'] + ['g']
    metadata['shape'] = (
        [len(data_source)]
        + metadata['shape'][:-1]
        + [prediction_window_length, 2]
    )
    pred_tensorstore_config = {
        'metadata': metadata
    } | config.tensorstore_config.to_dict()
    pred_tensorstore_config['kvstore']['path'] = video_ts_path
    logging.info('Video tensorstore path %s', video_ts_path)
    video_ts = ts.open(
        pred_tensorstore_config | {'delete_existing': True, 'create': True}
    ).result()
    # transpose to [batch, t, x, y, z, true + pred]
    video_ts = video_ts.transpose([0, 4, 1, 2, 3, 5])[
        :,
        :,
        data_source.x_indexer,  # pytype: disable=attribute-error
        data_source.y_indexer,  # pytype: disable=attribute-error
        data_source.z_indexer,  # pytype: disable=attribute-error
    ]
    video_writer = TsForecastWriter(video_ts, max_queue=config.write_queue)
  else:
    video_writer = BlankForecastWriter()

  if jax.process_index() == 0 and config.write_trace_forecast:
    num_traces = trace_mask.counts.shape[0] - 1  # without background
    # last dimension is for ground truth and forecast, respectively
    trace_config = {
        'dtype': 'float32',
        'rank': 4,
        'metadata': {
            'shape': [
                len(data_source),
                num_traces,
                prediction_window_length,
                2,
            ],
            'chunk_grid': {
                'name': 'regular',
                'configuration': {'chunk_shape': [1, num_traces, 1, 1]},
            },
        },
        'create': True,
        'delete_existing': True,
    } | config.tensorstore_config.to_dict()
    trace_ts_path = video_ts_path.replace('frames', 'traces')
    logging.info('Video tensorstore path %s', trace_ts_path)
    trace_config['kvstore']['path'] = trace_ts_path  # pytype: disable=unsupported-operands
    # writing always [timesteps, num_traces]
    trace_ts = ts.open(trace_config).result().transpose([0, 2, 1, 3])
    trace_writer = TsForecastWriter(trace_ts, max_queue=config.write_queue)
  else:
    trace_writer = BlankForecastWriter()

  cpoint_dir = f'{exp_workdir}/checkpoints'
  exp_config.init_from_cpoint = f'{cpoint_dir}-0/ckpt-{config.cpoint_id}'
  if config.cpoint_id > 0:
    # invalid but existing directory so latest cannot be loaded and defaults
    # to the config.init_from_cpoint checkpoint instead. This is due to the
    # logic of train.init_model.
    cpoint_dir = '/'.join(cpoint_dir.split('/')[:-1])
  exp_config = ml_collections.FrozenConfigDict(exp_config)
  model, state, _, _, _ = train.init_model(
      exp_config, rng, cpoint_dir, data_source.item_shape
  )

  # sync because otherwise the job finishes without writing xmanager metrics.
  writer = metric_writers.create_default_writer(
      infer_workdir, just_logging=jax.process_index() > 0, asynchronous=False
  )
  writer.write_hparams({
      k: v for k, v in config.items() if isinstance(v, (bool, float, int, str))
  })
  report_progress = training.ReportProgress(
      batch_size=timesteps_output,
      num_train_steps=len(data_source) * num_pred_steps,
      writer=writer,
  )

  temporal_metrics = {i: Metrics.empty() for i in range(num_pred_steps)}

  def replicate(arr):
    addressable_devices = replicate_sharding.addressable_devices
    arrs = [jax.device_put(arr, d) for d in addressable_devices]
    return jax.make_array_from_single_device_arrays(
        arr.shape, replicate_sharding, arrs
    )

  def to_global_array(sample):
    return {k: v.to_global() for k, v in sample.items()}  # pytype: disable=attribute-error

  state = jax.tree.map(replicate, state)
  trace_mask = jax.tree.map(replicate, trace_mask)
  loss_mask = jax.tree.map(
      replicate, train.get_mask(exp_config.loss_mask, masks)
  )
  input_mask = jax.tree.map(
      replicate,
      train.get_mask(
          exp_config.input_mask
          if hasattr(exp_config, 'input_mask')
          else 'none',
          masks,
          subsample=True,
      ),
  )

  def pred_fn(state, sample):
    if exp_config.eval_with_train_step:
      logging.info('Evaluating with train step')
      pred_fn = predict_bw
    else:
      logging.info('Evaluating with eval step')
      pred_fn = predict
    return pred_fn(
        model,
        state,
        sample,
        exp_config,
        trace_mask,
        loss_mask,
        input_mask,
        frame_sharding,
        stim_sharding,
        config.mask_video,
    )

  in_shardings = (
      replicate_sharding,  # params
      sample_sharding,  # inputs
  )
  out_sharding = (
      frame_sharding,  # predictions
      replicate_sharding,  # trace predictions
      replicate_sharding,  # trace ground truth
      replicate_sharding,  # metrics
      replicate_sharding,  # params/grad
  )

  s_predict = jax.jit(pred_fn, in_shardings, out_sharding)
  logging.info('Prepared model, replicated, and start inference.')

  with metric_writers.ensure_flushes(writer):
    for t, batch in enumerate(loader):
      logging.info('Inference for batch %i', t + 1)
      batch = to_global_array(batch)
      for i in range(0, prediction_window_length, timesteps_output):
        logging.info('Inference for step %i', i + 1)
        output_frames = batch['output_frames'][:, i : i + timesteps_output]
        output_stimulus = batch['output_stimulus'][:, i : i + timesteps_output]
        if sample_lead_time:
          batch['lead_time'] = replicate(np.array([i + 1], dtype=np.float32))

        sample = {
            'input_frames': batch['input_frames'],
            'input_stimulus': batch['input_stimulus'],
            'output_frames': output_frames,
            'output_stimulus': output_stimulus,
            'lead_time': batch['lead_time'],
        }
        with report_progress.timed('predict'):
          pred_frames, pred_trace, target_trace, metrics_update = s_predict(
              state, sample
          )[:4]
        logging.info('Global predictions shape %r', pred_frames.shape)
        logging.info('Trace predictions shape %r', pred_trace.shape)

        with jax.spmd_mode('allow_all'):
          gather = multihost_utils.process_allgather
          t_slice = slice(i, i + timesteps_output)
          if t % config.write_video_frequency == 0:
            # remove batch and channel dimensions for video
            video_writer.write_true(
                t, t_slice, gather(output_frames[0, ..., 0])
            )
            video_writer.write_pred(t, t_slice, gather(pred_frames[0, ..., 0]))
          if t % config.write_trace_frequency == 0:
            # remove batch dimension
            target_trace_out = gather(target_trace[0])
            pred_trace_out = gather(pred_trace[0])

            # TODO(jan-matthis,mjanusz): confirm that this always holds
            assert target_trace_out.shape[0] == 1
            assert pred_trace_out.shape[0] == 1
            target_trace_out = target_trace_out[0]
            pred_trace_out = pred_trace_out[0]

            trace_writer.write_true(t, t_slice, target_trace_out)
            trace_writer.write_pred(t, t_slice, pred_trace_out)
          temporal_metrics[i] = temporal_metrics[i].merge(metrics_update)
        if not sample_lead_time and num_pred_steps > 1:
          # combine forecast with input for autoregressive forecast
          sample['input_frames'] = combine_input_and_prediction(
              sample['input_frames'], pred_frames, exp_config
          )
          sample['input_stimulus'] = combine_input_and_prediction(
              sample['input_stimulus'], output_stimulus, exp_config
          )
        if jax.process_index() == 0:
          report_progress(t * num_pred_steps + i)

    if jax.process_index() == 0:
      with jax.spmd_mode('allow_all'):
        write_metrics(
            writer, temporal_metrics, num_pred_steps, timesteps_output
        )
        if config.infer_save_json:
          ts_utils.write_json(
              to_write=metrics_to_dict(
                  temporal_metrics, num_pred_steps, timesteps_output
              ),
              kvstore=os.path.join(
                  config.json_path_prefix.format(
                      workdir=infer_workdir,
                      base_path=config.base_path,
                      xm_id=config.xm_id,
                      work_unit=config.work_unit,
                      cpoint_id=config.cpoint_id,
                  ),
                  f'{config.split}_condition_{config.condition}.json'
              ),
          )

    trace_writer.flush()
    video_writer.flush()
