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

"""Data source backed by TensorStore."""

import dataclasses
from typing import Any, Sequence

from connectomics.jax import grain_util
import grain.python as grain
import tensorstore as ts

FlatFeatures = grain_util.FlatFeatures


@dataclasses.dataclass
class TensorStoreTimeSeriesConfig:
  """Config settings for TensorStoreTimeSeries.

  Attributes:
    input_spec: Spec for TensorStore with shape TxF (timesteps x features).
    timesteps_input: Number of timesteps for input arrays.
    timesteps_output: Number of timesteps for output arrays.
  """

  input_spec: dict[str, Any]
  timesteps_input: int
  timesteps_output: int


class TensorStoreTimeSeries:
  """TensorStore data source for time-series prediction."""

  def __init__(
      self,
      config: TensorStoreTimeSeriesConfig,
      transforms: Sequence[grain.MapTransform] = (),
      prefetch: bool = False,
      prefix: str = 'series',
      sequential: bool = True,
  ):
    """TensorStoreTimeSeries.

    Args:
      config: TensorStoreTimeSeriesConfig.
      transforms: Optional list of transforms applied on records.
      prefetch: If True, pulls all data and stores it in NumPy array.
      prefix: Prefix for dictionary keys.
      sequential: If True, output follows input without overlap. Otherwise,
        output is a one-step ahead shifted version of input.
    """
    if not sequential and config.timesteps_input != config.timesteps_output:
      raise ValueError(
          'Input and output timesteps must be equal for non-sequential source.')

    self.volume = ts.open(config.input_spec).result()

    if prefetch:
      self.array = self.volume.read().result()

    if sequential:
      offset = config.timesteps_input + config.timesteps_output
    else:
      offset = config.timesteps_input + 1
    self._len = self.volume.shape[0] - offset + 1
    assert self._len > 0, 'Dataset too small for timestep settings.'

    _, self.n_len = self.volume.shape
    self.n_indexer = slice(0, self.n_len)

    self.t_in = config.timesteps_input
    self.t_out = config.timesteps_output

    self.config = config
    self.transforms = transforms
    self.prefetch = prefetch
    self.prefix = prefix
    self.sequential = sequential

  def __len__(self) -> int:
    """Number of items in the dataset."""
    return self._len

  def _apply_transforms(self, features: FlatFeatures) -> FlatFeatures:
    for transform in self.transforms:
      # TODO(jan-matthis): Consider supporting other transforms as needed.
      features = transform.map(features)
    return features

  def __getitem__(self, record_key: int) -> FlatFeatures:
    """Fetch the items with the given record keys using Tensorstore.

    Args:
      record_key: time step integer index

    Returns:
      Dictionary with keys `timestep`, `{prefix}_input`, and `{prefix}_output`,
      where `{prefix}` is replaced according to the config.

    Raises:
      IndexError: when record_key out of bounds [0, self._len)
    """
    if record_key < 0 or record_key >= self._len:
      raise IndexError('Index out of bounds.')
    t_indexer_input = slice(record_key, record_key + self.t_in)
    if self.sequential:
      out_start = record_key + self.t_in
    else:
      out_start = record_key + 1
    t_indexer_output = slice(out_start, out_start + self.t_out)
    if not self.prefetch:
      input_array = self.volume[t_indexer_input, self.n_indexer].read().result()
      output_array = (
          self.volume[t_indexer_output, self.n_indexer].read().result()
      )
    else:
      input_array = self.array[t_indexer_input, self.n_indexer]
      output_array = self.array[t_indexer_output, self.n_indexer]
    return self._apply_transforms({
        'timestep': record_key,
        f'{self.prefix}_input': input_array,
        f'{self.prefix}_output': output_array,
    })

  def __repr__(self) -> str:
    return (
        f'TensorStoreTimeSeries(config={self.config}, prefetch={self.prefetch},'
        + f' transforms={self.transforms!r}), sequential={self.sequential}'
    )

  @property
  def item_shape(self) -> dict[str, tuple[int, ...]]:
    return {
        'timestep': tuple(),
        f'{self.prefix}_input': (self.t_in, self.n_len,),
        f'{self.prefix}_output': (self.t_out, self.n_len,),
    }


class MergedTensorStoreTimeSeries:
  """MergedTensorStoreTimeSeries."""

  def __init__(self, *srcs):
    """MergedTensorStoreTimeSeries.

    Args:
      *srcs: TensorStoreTimeSeries sources to merge; need to have same length.
    """
    self.srcs = list(srcs)
    for src in self.srcs:
      assert len(src) == len(self.srcs[0])

  def __len__(self):
    return len(self.srcs[0])

  def __getitem__(self, record_key: int) -> FlatFeatures:
    out = dict()
    for s in self.srcs:
      out.update(s[record_key])
    return out

  def __repr__(self) -> str:
    return (
        'MergedTensorStoreTimeSeries('
        + ','.join([f'{s}' for s in self.srcs])
        + ')'
    )

  @property
  def item_shape(self) -> dict[str, tuple[int, ...]]:
    out = dict()
    for s in self.srcs:
      out.update(s.item_shape)
    return out


class ConcatenatedTensorStoreTimeSeries:
  """ConcatenatedTensorStoreTimeSeries."""

  def __init__(self, *srcs):
    """ConcatenatedTensorStoreTimeSeries.

    Args:
      *srcs: TensorStoreTimeSeries sources to concatenate; need to have same
        item shape.
    """
    self.srcs = list(srcs)

    # Using a dictionary for lookup of source and local record key.
    # Consider using another data structure for very long time series.
    self.record_key_lookup = {}

    record_key = 0
    for src_idx, src in enumerate(self.srcs):
      assert src.item_shape == self.srcs[0].item_shape
      for local_record_key in range(len(src)):
        self.record_key_lookup[record_key] = (src_idx, local_record_key)
        record_key += 1
    self.max_record_key = record_key - 1

  def __len__(self):
    return self.max_record_key + 1

  def __getitem__(self, record_key: int) -> FlatFeatures:
    src_idx, local_record_key = self.record_key_lookup[record_key]
    return self.srcs[src_idx][local_record_key]

  def __repr__(self) -> str:
    return (
        'ConcatenatedTensorStoreTimeSeries('
        + ','.join([f'{s}' for s in self.srcs])
        + ')'
    )

  @property
  def item_shape(self) -> dict[str, tuple[int, ...]]:
    return self.srcs[0].item_shape
