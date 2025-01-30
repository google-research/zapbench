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

"""Tests for data utilities."""

import tensorstore as ts

from absl.testing import absltest
from absl.testing import parameterized
from zapbench import constants
from zapbench import data_utils


class DataTest(parameterized.TestCase):

  def test_get_spec(self):
    for spec_name in constants.SPECS:
      spec = data_utils.get_spec(spec_name)
      self.assertIsInstance(spec, ts.Spec)

  def test_get_covariate_spec(self):
    for spec_name in constants.COVARIATE_SPECS:
      spec = data_utils.get_covariate_spec(spec_name)
      self.assertIsInstance(spec, ts.Spec)

  def test_get_position_embedding_spec(self):
    for spec_name in constants.POSITION_EMBEDDING_SPECS:
      spec = data_utils.get_position_embedding_spec(spec_name)
      self.assertIsInstance(spec, ts.Spec)

  def test_get_rastermap_spec(self):
    for spec_name in constants.RASTERMAP_SPECS:
      spec = data_utils.get_rastermap_spec(spec_name)
      self.assertIsInstance(spec, ts.Spec)

  def test_condition_bounds(self):
    for condition in range(9):
      inclusive_min, exclusive_max = data_utils.get_condition_bounds(condition)
      self.assertLess(inclusive_min, exclusive_max)
    self.assertRaises(ValueError, data_utils.get_condition_bounds, -1)
    self.assertRaises(ValueError, data_utils.get_condition_bounds, 9)

  def test_adjust_condition_bounds_for_split(self):
    inclusive_min, exclusive_max = data_utils.adjust_condition_bounds_for_split(
        'train', 1, 1001, 12
    )
    self.assertEqual(inclusive_min, 1)
    self.assertEqual(exclusive_max, 700 + 1)

    inclusive_min, exclusive_max = data_utils.adjust_condition_bounds_for_split(
        'val', 1, 1001, 12
    )
    self.assertEqual(inclusive_min, 1 + 700 - 12)
    self.assertEqual(exclusive_max, 800 + 1)

    inclusive_min, exclusive_max = data_utils.adjust_condition_bounds_for_split(
        'test', 1, 1001, 12
    )
    self.assertEqual(inclusive_min, 1 + 800 - 12)
    self.assertEqual(exclusive_max, 1000 + 1)

    inclusive_min, exclusive_max = data_utils.adjust_condition_bounds_for_split(
        'test_holdout', 1, 1001, 12
    )
    self.assertEqual(inclusive_min, 1 + 256 - 12)
    self.assertEqual(exclusive_max, 1000 + 1)

    inclusive_min, exclusive_max = data_utils.adjust_condition_bounds_for_split(
        None, 1, 1001, 12
    )
    self.assertEqual(inclusive_min, 1)
    self.assertEqual(exclusive_max, 1000 + 1)

  def test_get_num_windows(self):
    num_windows = data_utils.get_num_windows(
        inclusive_min=1, exclusive_max=32 + 12 + 1, num_timesteps_context=12
    )
    self.assertEqual(num_windows, 1)

  @parameterized.parameters('train', 'val', 'test', 'test_holdout')
  def test_adjust_spec_for_condition_and_split(self, split):
    for spec_name in constants.SPECS:
      spec = data_utils.adjust_spec_for_condition_and_split(
          data_utils.get_spec(spec_name), 1, split, 12
      )
      self.assertIsInstance(spec, ts.Spec)


if __name__ == '__main__':
  absltest.main()
