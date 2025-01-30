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

"""Data sources and loaders based on pygrain using tensorstore."""

import dataclasses
from typing import Any, Sequence

from absl import logging
import chex
from connectomics.jax.inputs import tensorloader as tl
from connectomics.segmentation import labels
import flax
import jax
import ml_collections
import numpy as np
import tensorstore as ts

from zapbench import constants
from zapbench import data_utils


@dataclasses.dataclass(slots=True)
class LocalArray:
  """List of host local arrays with corresponding devices and global position."""

  arrays: Sequence[Any]
  devices: Sequence[jax.Device]
  sharding: jax.sharding.NamedSharding
  global_shape: tuple[int, ...]

  def result(self):
    def materialize_if_necessary(array: Any) -> Any:
      if hasattr(array, 'result'):
        return array.result()
      return array

    self.arrays = [materialize_if_necessary(arr) for arr in self.arrays]
    return self

  def to_global(self) -> chex.Array:
    arrays = [jax.device_put(a, d) for a, d in zip(self.arrays, self.devices)]
    global_array = jax.make_array_from_single_device_arrays(
        self.global_shape, self.sharding, arrays
    )
    return global_array


@dataclasses.dataclass(slots=True)
class GlobalFuture:
  """Global tensorstore array that is not read or materialized."""

  array: Any
  sharding: jax.sharding.NamedSharding

  def read(self) -> LocalArray:
    """Shard indexable array and read local array shards."""

    def read_if_necessary(array: Any) -> Any:
      if hasattr(array, 'read'):
        return array.read()
      return array

    device_arrays = [
        (d, read_if_necessary(self.array[index]))
        for d, index in self.sharding.addressable_devices_indices_map(
            self.array.shape
        ).items()
    ]
    devices = [da[0] for da in device_arrays]
    arrays = [da[1] for da in device_arrays]
    return LocalArray(arrays, devices, self.sharding, self.array.shape)


def get_timesteps_for_split(
    config: ml_collections.ConfigDict, split: str, sample_lead_time: bool
) -> tuple[Sequence[int], Sequence[int]]:
  """Get timesteps for a given split (train, val, test) respecting pre-training."""
  if sample_lead_time:
    timesteps_output = constants.PREDICTION_WINDOW_LENGTH
  else:
    timesteps_output = config.timesteps_output
  model_offset = config.timesteps_input + timesteps_output
  condition_offsets = config.condition_offsets
  timesteps = []
  boundaries = []
  train_splits = ['train', 'train_val']
  # Iterate through conditions and extend timesteps based on split.
  for condition in config.conditions:
    if condition in constants.CONDITIONS_TRAIN and split != 'test_holdout':
      # in Fish 2.0 train/val/test conditions
      t_start, t_end = data_utils.adjust_condition_bounds_for_split(
          split,
          *data_utils.get_condition_bounds(condition),
          config.timesteps_input,
      )
    elif condition in constants.CONDITIONS_HOLDOUT and split == 'test_holdout':
      # in Fish 2.0 holdout conditions
      t_start, t_end = data_utils.adjust_condition_bounds_for_split(
          split,
          *data_utils.get_condition_bounds(condition),
          config.timesteps_input,
      )
    elif condition not in constants.CONDITIONS and split in train_splits:
      # pretraining conditions are only used for training, use config offsets
      t_start = condition_offsets[condition] + constants.CONDITION_PADDING
      t_end = condition_offsets[condition + 1] - constants.CONDITION_PADDING
    else:
      continue
    timesteps.extend(range(t_start, t_end - model_offset + 1))
    num_timesteps = t_end - model_offset + 1 - t_start
    boundaries.extend(num_timesteps * [t_end])
  return timesteps, boundaries


class VideoTensorSource(tl.TensorSource):
  """Tensorstore data source for video prediction task.

  The data are read from tensorstore using the open config in
  config.tensorstore_config and assumes a shape of [x, y, z, t] where t are the
  timesteps and therefore the sample dimension for video prediction. The splits
  are based on benchmark data utils with extension for pretraining volumes.

  The data source indexes and operates on self.timesteps, which denote all valid
  starting indices.
  """

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      frame_sharding: jax.sharding.NamedSharding,
      stim_sharding: jax.sharding.NamedSharding,
      lead_time_sharding: jax.sharding.NamedSharding,
      split: str = 'train',
      sample_lead_time: bool = False,
  ):
    assert split in ['train', 'val', 'train_val', 'test', 'test_holdout']
    self.config = config
    in_volume = ts.open(config.tensorstore_input_config.to_dict()).result()
    out_volume = ts.open(config.tensorstore_output_config.to_dict()).result()
    self.in_volume = in_volume.transpose([3, 0, 1, 2])  # t, x_in, y_in, z_in
    self.out_volume = out_volume.transpose([3, 0, 1, 2])  # t, x, y, z

    ts_stimulus_config_dict = config.tensorstore_stimulus_config.to_dict()
    self.stim_volume = ts.open(ts_stimulus_config_dict).result()  # t, d

    if self.in_volume.shape[0] != self.out_volume.shape[0]:
      raise ValueError('Timesteps of in and output volume do not match.')
    if self.in_volume.shape[0] != self.stim_volume.shape[0]:
      raise ValueError('Timesteps of volume and stimulus do not match.')

    self.timesteps, self.boundaries = get_timesteps_for_split(
        config, split, sample_lead_time
    )
    self.sample_lead_time = sample_lead_time
    self.split = split

    # slice out center at full resolution based on given crop sizes
    _, x_size, y_size, z_size = self.out_volume.shape
    x_crop_size = x_size if config.x_crop_size is None else config.x_crop_size
    y_crop_size = y_size if config.y_crop_size is None else config.y_crop_size
    z_crop_size = z_size if config.z_crop_size is None else config.z_crop_size
    x_start = max((x_size - x_crop_size) // 2, 0)
    x_end = min(x_start + x_crop_size, x_size)
    self.x_indexer = slice(x_start, x_end)
    self.x_len = x_end - x_start
    y_start = max((y_size - y_crop_size) // 2, 0)
    y_end = min(y_start + y_crop_size, y_size)
    self.y_indexer = slice(y_start, y_end)
    self.y_len = y_end - y_start
    z_start = max((z_size - z_crop_size) // 2, 0)
    z_end = min(z_start + z_crop_size, z_size)
    self.z_indexer = slice(z_start, z_end)
    self.z_len = z_end - z_start
    logging.info(
        'Full resolution indexers: %r, %r, %r',
        self.x_indexer,
        self.y_indexer,
        self.z_indexer,
    )

    # compute slices at input resolution
    _, x_size_in, y_size_in, z_size_in = self.in_volume.shape
    assert x_size_in <= x_size and y_size_in <= y_size and z_size_in <= z_size
    x_factor = x_size // x_size_in
    y_factor = y_size // y_size_in
    z_factor = z_size // z_size_in
    x_crop_size_in = x_crop_size // x_factor
    y_crop_size_in = y_crop_size // y_factor
    z_crop_size_in = z_crop_size // z_factor
    x_start = max((x_size_in - x_crop_size_in) // 2, 0)
    x_end = min(x_start + x_crop_size_in, x_size_in)
    x_indexer_in = slice(x_start, x_end)
    self.x_in = x_end - x_start
    y_start = max((y_size_in - y_crop_size_in) // 2, 0)
    y_end = min(y_start + y_crop_size_in, y_size_in)
    y_indexer_in = slice(y_start, y_end)
    self.y_in = y_end - y_start
    z_start = max((z_size_in - z_crop_size_in) // 2, 0)
    z_end = min(z_start + z_crop_size_in, z_size_in)
    z_indexer_in = slice(z_start, z_end)
    self.z_in = z_end - z_start
    logging.info(
        'Input resolution indexers: %r, %r, %r',
        x_indexer_in,
        y_indexer_in,
        z_indexer_in,
    )

    self.in_volume = self.in_volume[:, x_indexer_in, y_indexer_in, z_indexer_in]
    self.out_volume = self.out_volume[
        :, self.x_indexer, self.y_indexer, self.z_indexer
    ]
    self.t_in = config.timesteps_input
    self.t_out = config.timesteps_output
    self.frame_sharding = frame_sharding
    self.stim_sharding = stim_sharding
    self.lead_time_sharding = lead_time_sharding
    self.out_to_in_scale_xyz = (x_factor, y_factor, z_factor)

  def __len__(self) -> int:
    """Number of items in the dataset."""
    return len(self.timesteps)

  def check_indices(self, indices: Sequence[int]):
    indices_array = np.array(indices)
    if any((indices_array < 0) | (indices_array >= len(self))):
      raise IndexError('One or many indices out of bounds.')

  def __getitem__(self, metadata: tl.BatchMetadata | int) -> tl.Batch:
    """Fetch the items with the given record keys using Tensorstore.

    Args:
      metadata: index or indices and rng keys for the batch to retrieve.

    Returns:
      dictionary with keys 'input_frames', and 'output_frames' and corresponding
      stimuli. Each element has leading batch dimension for metadata.indices.

    Raises:
      IndexError: when record_key out of bounds [0, len(self))
    """
    if isinstance(metadata, int):
      metadata = tl.BatchMetadata(indices=[metadata])
    self.check_indices(metadata.indices)
    translated_indices = [self.timesteps[i] for i in metadata.indices]
    boundaries = [self.boundaries[i] for i in metadata.indices]
    in_ixs = [list(range(s, s + self.t_in)) for s in translated_indices]
    translated_indices = [s + self.t_in for s in translated_indices]
    if self.sample_lead_time:
      assert self.t_out == 1
      assert metadata.rngs is not None
      max_lead_time = constants.PREDICTION_WINDOW_LENGTH

      def randint(rng, low, high):
        if low == high:  # can only sample one value
          return low
        return rng.integers(low, high)

      # up to max_lead_time - 1 since we additionally sample from [0, 1] below
      shifted_indices = [
          randint(rng, s, min(s + max_lead_time, boundary))
          for rng, s, boundary in zip(
              metadata.rngs, translated_indices, boundaries
          )
      ]
      # offset between 1 and prediction_window_length (32)
      offsets = [
          shifted_index - index + 1
          for index, shifted_index in zip(translated_indices, shifted_indices)
      ]
      out_ixs = [[s] for s in shifted_indices]
    else:
      offsets = [0] * len(translated_indices)
      out_ixs = [list(range(s, s + self.t_out)) for s in translated_indices]
    in_frames = GlobalFuture(
        self.in_volume[in_ixs].translate_to[0][..., None], self.frame_sharding
    ).read()
    out_frames = GlobalFuture(
        self.out_volume[out_ixs].translate_to[0][..., None], self.frame_sharding
    ).read()
    in_stim = GlobalFuture(
        self.stim_volume[in_ixs].translate_to[0], self.stim_sharding
    ).read()
    out_stim = GlobalFuture(
        self.stim_volume[out_ixs].translate_to[0], self.stim_sharding
    ).read()
    lead_time = GlobalFuture(
        np.array(offsets).astype(np.float32), self.lead_time_sharding
    ).read()
    return dict(
        input_frames=in_frames.result(),
        output_frames=out_frames.result(),
        input_stimulus=in_stim.result(),
        output_stimulus=out_stim.result(),
        lead_time=lead_time.result(),
    )

  @property
  def item_shape(self) -> dict[str, tuple[int, ...]]:
    """Return shape of items returned unsharded."""
    return dict(
        input_frames=(self.t_in, self.x_in, self.y_in, self.z_in, 1),
        output_frames=(self.t_out, self.x_len, self.y_len, self.z_len, 1),
        input_stimulus=(self.t_in, self.stim_volume.shape[1]),
        output_stimulus=(self.t_out, self.stim_volume.shape[1]),
        lead_time=tuple(),
    )


class TraceMask(flax.struct.PyTreeNode):
  segment_ids: chex.Array
  counts: chex.Array


class Masks(flax.struct.PyTreeNode):
  trace_mask: TraceMask
  brain_mask: chex.Array
  out_to_in_scale_xyz: tuple[int, int, int]


def load_segmentation(
    config: ml_collections.ConfigDict,
) -> tuple[chex.Array, int]:
  """Load and prepare mask with individual traces or for all cells globally.

  Args:
    config: training configuration

  Returns:
    tuple(trace_mask, max_id) where trace mask associates each voxel to a trace
    id (0 is background) and max_id is the highest assigned trace id.
  """
  seg_store = ts.open(config.seg_tensorstore_config.to_dict()).result()
  seg = seg_store.read().result()
  max_id = np.max(seg)
  slices = [slice(f // 2, None, f) for f in config.seg_downsampling]
  return seg[*slices], max_id


def load_brain_mask(config: ml_collections.ConfigDict) -> chex.Array:
  """Load and prepare a mask that covers the brain region of Fish 2.0.

  Args:
    config: training configuration

  Returns:
    binary mask
  """
  mask_store = ts.open(config.mask_tensorstore_config.to_dict()).result()
  mask = mask_store.read().result()
  slices = [slice(f // 2, None, f) for f in config.mask_downsampling]
  return mask[*slices]


def get_dataset(
    config: ml_collections.ConfigDict,
    num_steps: int,
    frame_sharding: jax.sharding.NamedSharding,
    stim_sharding: jax.sharding.NamedSharding,
    lead_time_sharding: jax.sharding.NamedSharding,
    split: str = 'train',
    return_masks: bool = False,
    contiguous_segments: bool = True,
    sample_lead_time: bool = False,
    shuffle: bool = True,
) -> tl.TensorLoader | tuple[tl.TensorLoader, Masks]:
  """Get train or val dataset loaders, optionally load and return mask.

  Args:
    config: configuration
    num_steps: number of steps to take (length of data loader)
    frame_sharding: sharding of data batches [b, t, x, y, (z)]
    stim_sharding: sharding of stimulus data [b, t]
    lead_time_sharding: sharding of lead time [b,]
    split: {'train', 'val', 'train_val', 'test', 'test_holdout'} denoting split
    return_masks: whether to return tuple of mask and its segment counts
    contiguous_segments: whether segments should have contiguous ids or preserve
      all original trace ids.
    sample_lead_time: whether data loader should sample lead times
    shuffle: whether to shuffle the timesteps

  Returns:
    DataLoader and optionally a Mask
  """
  assert split in ['train', 'val', 'train_val', 'test', 'test_holdout']
  train = split in ['train', 'train_val']
  drop_remainder = train or not config.eval_pad_last_batch

  tsource = VideoTensorSource(
      config.data_config,
      frame_sharding=frame_sharding,
      stim_sharding=stim_sharding,
      lead_time_sharding=lead_time_sharding,
      split=split,
      sample_lead_time=sample_lead_time,
  )
  steps_per_epoch = len(tsource) // config.global_batch_size
  num_epochs = num_steps // steps_per_epoch + bool(num_steps % steps_per_epoch)
  data_loader = tl.TensorLoader(
      tensor_source=tsource,
      batch_size=config.global_batch_size,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=config.seed,
      shard_options=tl.ShardOptions(
          shard_index=0, shard_count=1, drop_remainder=drop_remainder
      ),
      num_threads=config.data_config.num_threads,
  )

  if return_masks:
    trace_segs, max_id = load_segmentation(config)
    trace_segs = trace_segs[
        tsource.x_indexer, tsource.y_indexer, tsource.z_indexer
    ]
    logging.info('Loaded trace seg. %r with %r cells', trace_segs.shape, max_id)
    # Contiguous is used for computing evaluation metrics since it restricts
    # to available traces and therefore does not divide by zero when computing
    # metrics. Non-contiguous is used when forecasting for traces while
    # maintaining the order of traces for later association and comparison.
    if contiguous_segments:
      # make trace ids contiguous and ignore those not present
      trace_segs, _ = labels.make_contiguous(trace_segs)
      _, trace_counts = np.unique(trace_segs.reshape(-1), return_counts=True)
      logging.info(
          'Modified trace mask to shape %r with %r unique segments remaining',
          trace_segs.shape,
          trace_counts.shape[0],
      )
    else:
      # maintain trace ids even if not present after resizing
      trace_counts = np.zeros(int(max_id) + 1, dtype=np.int32)
      ids, counts = np.unique(trace_segs.reshape(-1), return_counts=True)
      trace_counts[ids] = counts

    brain_mask = load_brain_mask(config)
    brain_mask = brain_mask[
        tsource.x_indexer, tsource.y_indexer, tsource.z_indexer
    ]
    trace_mask = TraceMask(trace_segs, trace_counts)
    mask = Masks(trace_mask, brain_mask, tsource.out_to_in_scale_xyz)
    return data_loader, mask

  return data_loader
