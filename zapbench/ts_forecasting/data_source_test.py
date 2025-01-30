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

"""Tests for data source."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from zapbench.ts_forecasting import data_source


class DataSourceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_timesteps, self.num_features = 32, 2
    self.default_config = data_source.TensorStoreTimeSeriesConfig(
        input_spec={
            'driver': 'array',
            'dtype': 'float32',
            'array': np.ones((self.num_timesteps, self.num_features)).tolist(),
        },
        timesteps_input=4,
        timesteps_output=2,
    )

  @parameterized.parameters(
      {'timesteps_input': 1, 'timesteps_output': 1, 'sequential': True},
      {'timesteps_input': 2, 'timesteps_output': 2, 'sequential': True},
      {'timesteps_input': 3, 'timesteps_output': 2, 'sequential': True},
      {'timesteps_input': 2, 'timesteps_output': 3, 'sequential': True},
      {'timesteps_input': 6, 'timesteps_output': 4, 'sequential': True},
      {'timesteps_input': 6, 'timesteps_output': 6, 'sequential': True},
      {'timesteps_input': 1, 'timesteps_output': 1, 'sequential': False},
      {'timesteps_input': 2, 'timesteps_output': 2, 'sequential': False},
      {'timesteps_input': 6, 'timesteps_output': 6, 'sequential': False},
  )
  def test_data_source_shape_with_different_timesteps(
      self, timesteps_input: int, timesteps_output: int, sequential: bool):
    ds = data_source.TensorStoreTimeSeries(
        config=data_source.TensorStoreTimeSeriesConfig(
            input_spec={
                'driver': 'array',
                'dtype': 'float32',
                'array': np.ones(
                    (self.num_timesteps, self.num_features)).tolist(),
            },
            timesteps_input=timesteps_input,
            timesteps_output=timesteps_output,
        ),
        sequential=sequential,
    )
    self.assertEqual(
        ds[0]['series_input'].shape, (timesteps_input, self.num_features))
    self.assertEqual(
        ds[0]['series_output'].shape, (timesteps_output, self.num_features))
    self.assertEqual(
        ds[0]['series_input'].shape, ds.item_shape['series_input'])
    self.assertEqual(
        ds[0]['series_output'].shape, ds.item_shape['series_output'])
    offset = (timesteps_input + timesteps_output) if sequential else (
        timesteps_input + 1)
    self.assertLen(ds, self.num_timesteps - offset + 1)

  @parameterized.parameters(
      {'timesteps_input': 1, 'timesteps_output': 1, 'sequential': True},
      {'timesteps_input': 2, 'timesteps_output': 2, 'sequential': True},
      {'timesteps_input': 3, 'timesteps_output': 2, 'sequential': True},
      {'timesteps_input': 2, 'timesteps_output': 3, 'sequential': True},
      {'timesteps_input': 6, 'timesteps_output': 4, 'sequential': True},
      {'timesteps_input': 6, 'timesteps_output': 6, 'sequential': True},
      {'timesteps_input': 1, 'timesteps_output': 1, 'sequential': False},
      {'timesteps_input': 2, 'timesteps_output': 2, 'sequential': False},
      {'timesteps_input': 6, 'timesteps_output': 6, 'sequential': False},
  )
  def test_data_source_values_with_different_timesteps(
      self, timesteps_input: int, timesteps_output: int, sequential: bool):
    ds = data_source.TensorStoreTimeSeries(
        config=data_source.TensorStoreTimeSeriesConfig(
            input_spec={
                'driver': 'array',
                'dtype': 'float32',
                'array': [
                    [j for _ in range(self.num_features)]
                    for j in range(self.num_timesteps)
                ],
            },
            timesteps_input=timesteps_input,
            timesteps_output=timesteps_output,
        ),
        sequential=sequential,
    )
    np.testing.assert_array_equal(
        ds[0]['series_input'],
        np.array([
            [j for _ in range(self.num_features)]
            for j in range(ds[0]['series_input'].shape[0])
        ]),
    )
    if sequential:
      np.testing.assert_array_equal(
          ds[0]['series_output'],
          np.array([
              [
                  j + ds[0]['series_input'].shape[0]
                  for _ in range(self.num_features)
              ]
              for j in range(ds[0]['series_output'].shape[0])
          ]),
      )
    else:
      np.testing.assert_array_equal(
          ds[0]['series_output'],
          np.array([
              [
                  j + 1
                  for _ in range(self.num_features)
              ]
              for j in range(ds[0]['series_output'].shape[0])
          ]),
      )

  def test_merged_data_source(self):
    ds1 = data_source.TensorStoreTimeSeries(self.default_config, prefix='ds1')
    ds2 = data_source.TensorStoreTimeSeries(self.default_config, prefix='ds2')
    ds = data_source.MergedTensorStoreTimeSeries(ds1, ds2)
    self.assertLen(ds, len(ds1))
    self.assertLen(ds, len(ds2))
    self.assertEqual(ds.item_shape['ds1_input'], ds1.item_shape['ds1_input'])
    self.assertEqual(ds.item_shape['ds2_input'], ds2.item_shape['ds2_input'])
    batch = ds[0]
    for k in ('ds1_input', 'ds1_input', 'ds2_input', 'ds2_output', 'timestep'):
      assert k in batch

  def test_concatenated_data_source(self):
    ds1 = data_source.TensorStoreTimeSeries(self.default_config)
    ds2 = data_source.TensorStoreTimeSeries(self.default_config)
    ds = data_source.ConcatenatedTensorStoreTimeSeries(ds1, ds2)
    self.assertLen(ds, len(ds1) + len(ds2))


if __name__ == '__main__':
  absltest.main()
