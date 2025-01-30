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

"""Tests for hyperparameter utilities."""

from absl.testing import absltest
from zapbench import hparam_utils


class TestHparamUtils(absltest.TestCase):

  def test_fixed(self):
    self.assertEqual(
        list(hparam_utils.fixed('a', 1, length=2)), [{'a': 1}, {'a': 1}]
    )

  def test_merge_dicts(self):
    self.assertEqual(
        hparam_utils.merge_dicts({'a': 1}, {'b': 2}), {'a': 1, 'b': 2}
    )
    self.assertEqual(hparam_utils.merge_dicts({'a': 1}, {'a': 2}), {'a': 2})
    self.assertEqual(
        hparam_utils.merge_dicts({'a': 1, 'b': 2}, {'b': 3, 'c': 4}),
        {'a': 1, 'b': 3, 'c': 4},
    )
    self.assertEqual(hparam_utils.merge_dicts(), {})

  def test_product(self):
    iterables = [
        hparam_utils.sweep('learning_rate', [1e-3, 1e-4]),
        hparam_utils.sweep('batch_size', [32, 64]),
    ]
    expected = [
        {'learning_rate': 1e-3, 'batch_size': 32},
        {'learning_rate': 1e-3, 'batch_size': 64},
        {'learning_rate': 1e-4, 'batch_size': 32},
        {'learning_rate': 1e-4, 'batch_size': 64},
    ]
    self.assertEqual(list(hparam_utils.product(iterables)), expected)

    # Test with more than two iterables
    iterables = [
        hparam_utils.sweep('a', [1, 2]),
        hparam_utils.sweep('b', [3, 4]),
        hparam_utils.sweep('c', [5, 6]),
    ]
    expected = [
        {'a': 1, 'b': 3, 'c': 5},
        {'a': 1, 'b': 3, 'c': 6},
        {'a': 1, 'b': 4, 'c': 5},
        {'a': 1, 'b': 4, 'c': 6},
        {'a': 2, 'b': 3, 'c': 5},
        {'a': 2, 'b': 3, 'c': 6},
        {'a': 2, 'b': 4, 'c': 5},
        {'a': 2, 'b': 4, 'c': 6},
    ]
    self.assertEqual(list(hparam_utils.product(iterables)), expected)

  def test_sweep(self):
    self.assertEqual(
        list(hparam_utils.sweep('a', [1, 2, 3])), [{'a': 1}, {'a': 2}, {'a': 3}]
    )
    self.assertEqual(
        list(hparam_utils.sweep('b', ['x', 'y'])), [{'b': 'x'}, {'b': 'y'}]
    )

  def test_zipit(self):
    iterables = [
        hparam_utils.sweep('learning_rate', [1e-3, 1e-4]),
        hparam_utils.sweep('batch_size', [32, 64]),
    ]
    expected = [
        {'learning_rate': 1e-3, 'batch_size': 32},
        {'learning_rate': 1e-4, 'batch_size': 64},
    ]
    self.assertEqual(list(hparam_utils.zipit(iterables)), expected)

    # Test with more than two iterables
    iterables = [
        hparam_utils.sweep('a', [1, 2]),
        hparam_utils.sweep('b', [3, 4]),
        hparam_utils.sweep('c', [5, 6]),
    ]
    expected = [
        {'a': 1, 'b': 3, 'c': 5},
        {'a': 2, 'b': 4, 'c': 6},
    ]
    self.assertEqual(list(hparam_utils.zipit(iterables)), expected)


if __name__ == '__main__':
  absltest.main()
