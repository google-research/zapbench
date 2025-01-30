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

"""Tests for util."""

import time

from absl.testing import absltest
from absl.testing import parameterized
import distrax
import jax.numpy as jnp
import numpy as np

from zapbench.ts_forecasting import util


class UtilsTest(parameterized.TestCase):

  def test_max_runtime_stopping(self):
    mrs = util.MaxRuntimeStopping(max_runtime=0.5).set_reference_timestamp()

    tic = time.time()
    for _ in range(10):
      time.sleep(0.1)
      mrs = mrs.update()
      if mrs.should_stop:
        break
    toc = time.time()

    self.assertGreaterEqual((toc - tic), 0.5)
    self.assertLessEqual((toc - tic), 0.6)

  def test_track_best_step(self):
    tbs = util.TrackBestStep()
    tbs = tbs.update(metric=1.0, step=1)
    tbs = tbs.update(metric=0.5, step=2)
    tbs = tbs.update(metric=0.6, step=3)
    self.assertEqual(tbs.best_step, 2)
    self.assertEqual(tbs.best_metric, 0.5)

  def test_bin_upper_bounds(self):
    lower, upper, num_bins = 0.0, 1.0, 10
    np.testing.assert_allclose(
        util.get_bin_upper_bounds(
            lower=lower,
            upper=upper,
            num_bins=num_bins,
            extend_upper_interval=False,
        ),
        jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )
    np.testing.assert_allclose(
        util.get_bin_upper_bounds(
            lower=lower,
            upper=upper,
            num_bins=num_bins,
            extend_upper_interval=True,
        ),
        jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, jnp.inf]),
    )

  def test_digitize_bijector_forward(self):
    lower, upper, num_classes = 0.0, 1.0, 10
    array = jnp.array(
        [0.001, 0.099, 0.100, 0.101, 0.199, 0.899, 0.901, 2.000, -2.000],
        dtype=jnp.float32,
    )
    expected = jnp.array(
        [0.000, 0.000, 1.000, 1.000, 1.000, 8.000, 9.000, 9.000, +0.000],
        dtype=jnp.float32,
    )
    bij = util.get_digitize_bijector(
        lower=lower, upper=upper, num_classes=num_classes
    )
    np.testing.assert_allclose(bij.forward(array), expected)

  def test_digitize_bijector_inverse(self):
    lower, upper, num_classes = 0.0, 1.0, 10
    array = jnp.array([0.00, 1.00, 8.00, 9.00], dtype=jnp.float32)
    expected = jnp.array([0.05, 0.15, 0.85, 0.95], dtype=jnp.float32)
    bij = util.get_digitize_bijector(
        lower=lower, upper=upper, num_classes=num_classes
    )
    np.testing.assert_allclose(bij.inverse(array), expected)

  def test_one_hot_bijector_forward(self):
    num_classes = 10
    array = jnp.array([0.0, 1.0, 8.0, 9.0], dtype=jnp.float32)
    expected = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    bij = util.get_one_hot_bijector(num_classes=num_classes)
    np.testing.assert_allclose(bij.forward(array), expected)

  def test_one_hot_bijector_inverse(self):
    num_classes = 10
    array = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    expected = jnp.array([0.0, 1.0, 8.0, 9.0], dtype=jnp.float32)
    bij = util.get_one_hot_bijector(num_classes=num_classes)
    np.testing.assert_allclose(bij.inverse(array), expected)

  def test_digitize_and_one_hot_encode_forward(self):
    lower, upper, num_classes = 0.0, 1.0, 10
    array = jnp.array([0.05, 0.55, 0.95, -2.0, 2.0], dtype=jnp.float32)
    expected = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    chain = distrax.Chain(
        [
            util.get_digitize_bijector(
                lower=lower, upper=upper, num_classes=num_classes
            ),
            util.get_one_hot_bijector(num_classes=num_classes),
        ][::-1]
    )
    np.testing.assert_allclose(chain.forward(array), expected)

  def test_digitize_and_one_hot_encode_inverse(self):
    lower, upper, num_classes = 0.0, 1.0, 10
    array = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    expected = jnp.array([0.05, 0.55, 0.95, 0.05, 0.95], dtype=jnp.float32)
    chain = distrax.Chain(
        [
            util.get_digitize_bijector(
                lower=lower, upper=upper, num_classes=num_classes
            ),
            util.get_one_hot_bijector(num_classes=num_classes),
        ][::-1]
    )
    np.testing.assert_allclose(distrax.Inverse(chain).forward(array), expected)

  def test_get_metrics_collection(self):
    outputs = dict(
        learning_rate=0.01,
        loss=jnp.array(1.7),
        predictions=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        targets=jnp.array([[0.0, 2.0], [3.0, 4.0]]),
    )
    collection = util.get_metrics_collection(
        ['learning_rate', 'loss', 'mse', 'mse_step'], prefix='test_'
    ).single_from_model_output(**outputs)
    self.assertEqual(
        collection.test_learning_rate.value, outputs['learning_rate']
    )
    self.assertEqual(collection.test_loss.count, 1)
    self.assertEqual(collection.test_mse.count, 2)
    self.assertEqual(collection.test_mse_step.count, 2)

  @parameterized.parameters(['n5', 'zarr'])
  def test_save_and_load_array(self, driver):
    array = jnp.zeros((2, 2), dtype=jnp.float32)
    path = f'file://{self.create_tempdir().full_path}'
    util.save_array(array, path, driver=driver)
    loaded_array = util.load_array(path, driver=driver)
    np.testing.assert_equal(array, loaded_array)

  def test_get_condition_number_from_string(self):
    assert util.get_condition_number_from_string('test_condition_1.json'), 1
    assert util.get_condition_number_from_string('condition_12'), 12


if __name__ == '__main__':
  absltest.main()
