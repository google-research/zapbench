# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prefetching must not expose writable views of the cached time series."""

import grain.python as grain
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from zapbench.ts_forecasting import data_source


class InPlaceTransform(grain.MapTransform):

  def map(self, features):
    features['series_input'] += 100
    features['series_output'] *= 2
    return features


class PrefetchTest(parameterized.TestCase):

  def make_source(
      self, prefetch, sequential, offset=0, transforms=(), dtype='float32'
  ):
    self.values = np.arange(60, dtype=dtype).reshape(20, 3)
    source = data_source.TensorStoreTimeSeries(
        data_source.TensorStoreTimeSeriesConfig(
            input_spec={
                'driver': 'array',
                'dtype': dtype,
                'array': self.values.tolist(),
            },
            timesteps_input=4,
            timesteps_output=2 if sequential else 4,
            timesteps_output_offset=offset,
        ),
        prefetch=prefetch,
        sequential=sequential,
        transforms=transforms,
    )
    return source

  def expected(self, source, key):
    start = key + (source.t_in if source.sequential else 1)
    start += source.t_out_offset
    return (
        self.values[key : key + source.t_in],
        self.values[start : start + source.t_out],
    )

  @parameterized.product(
      prefetch=[False, True], sequential=[False, True], offset=[0, 2]
  )
  def test_mutating_a_record_does_not_change_cached_or_future_data(
      self, prefetch, sequential, offset
  ):
    source = self.make_source(prefetch, sequential, offset)
    record = source[1]
    expected_input, expected_output = self.expected(source, 1)
    np.testing.assert_array_equal(record['series_input'], expected_input)
    np.testing.assert_array_equal(record['series_output'], expected_output)
    record['series_input'] += 1000
    np.testing.assert_array_equal(record['series_output'], expected_output)
    record['series_output'] *= -1
    for key in (1, 0, 2, len(source) - 1):
      actual = source[key]
      expected_input, expected_output = self.expected(source, key)
      np.testing.assert_array_equal(actual['series_input'], expected_input)
      np.testing.assert_array_equal(actual['series_output'], expected_output)
    np.testing.assert_array_equal(source.volume.read().result(), self.values)
    if prefetch:
      np.testing.assert_array_equal(source.array, self.values)

  @parameterized.product(
      prefetch=[False, True],
      sequential=[False, True],
      dtype=['float32', 'int32'],
  )
  def test_in_place_transforms_only_change_the_current_record(
      self, prefetch, sequential, dtype
  ):
    source = self.make_source(
        prefetch, sequential, transforms=(InPlaceTransform(),), dtype=dtype
    )
    for key in (0, 1, 0, len(source) - 1, 1):
      record = source[key]
      expected_input, expected_output = self.expected(source, key)
      np.testing.assert_array_equal(
          record['series_input'], expected_input + 100
      )
      np.testing.assert_array_equal(
          record['series_output'], expected_output * 2
      )
      self.assertEqual(record['series_input'].dtype, np.dtype(dtype))
      self.assertEqual(record['series_output'].dtype, np.dtype(dtype))
    if prefetch:
      np.testing.assert_array_equal(source.array, self.values)

  @parameterized.product(prefetch=[False, True], sequential=[False, True])
  def test_grain_loader_repeated_epochs_preserve_samples(
      self, prefetch, sequential
  ):
    source = self.make_source(prefetch, sequential)
    loader = grain.DataLoader(
        data_source=source,
        sampler=grain.IndexSampler(
            num_records=len(source),
            num_epochs=2,
            shuffle=False,
            shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        ),
        operations=[InPlaceTransform()],
        worker_count=0,
    )
    records = list(loader)
    self.assertLen(records, 2 * len(source))
    for position, record in enumerate(records):
      key = position % len(source)
      expected_input, expected_output = self.expected(source, key)
      self.assertEqual(record['timestep'], key)
      np.testing.assert_array_equal(
          record['series_input'], expected_input + 100
      )
      np.testing.assert_array_equal(
          record['series_output'], expected_output * 2
      )

  @parameterized.parameters(False, True)
  def test_invalid_indices_still_raise(self, prefetch):
    source = self.make_source(prefetch, sequential=True)
    for key in (-1, len(source)):
      with self.assertRaises(IndexError):
        _ = source[key]


if __name__ == '__main__':
  absltest.main()
