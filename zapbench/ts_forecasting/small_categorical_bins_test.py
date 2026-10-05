# Copyright 2026 The Google Research Authors.
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

"""Finite decoding regressions for one- and two-class forecast heads."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from zapbench.ts_forecasting import heads
from zapbench.ts_forecasting import util


class SmallCategoricalBinsTest(parameterized.TestCase):

  @parameterized.product(
      num_classes=[1, 2, 3, 10, 64], bounds=[(0.0, 1.0), (-2.0, 3.0)]
  )
  def test_inverse_decodes_to_finite_bin_centers(self, num_classes, bounds):
    lower, upper = bounds
    bijector = util.get_digitize_bijector(lower, upper, num_classes)
    classes = jnp.arange(num_classes, dtype=jnp.float32)
    expected = (
        lower + (np.arange(num_classes) + 0.5) * (upper - lower) / num_classes
    )
    for inverse in (bijector.inverse, jax.jit(bijector.inverse)):
      decoded = inverse(classes)
      self.assertTrue(bool(jnp.isfinite(decoded).all()))
      np.testing.assert_allclose(decoded, expected, rtol=1e-6, atol=2e-7)
      np.testing.assert_array_equal(bijector.forward(decoded), classes)

  @parameterized.parameters(1, 2)
  def test_forward_tail_assignment_is_unchanged(self, num_classes):
    bijector = util.get_digitize_bijector(0.0, 1.0, num_classes)
    samples = jnp.array([-10.0, 0.0, 0.49, 0.5, 0.99, 1.0, 10.0])
    expected = np.zeros(7) if num_classes == 1 else [0, 0, 0, 1, 1, 1, 1]
    np.testing.assert_array_equal(bijector.forward(samples), expected)

  @parameterized.parameters(1, 2, 10)
  def test_actual_head_modes_samples_and_losses_remain_finite(
      self, num_classes
  ):
    head = heads.CategoricalHead(lower=-2.0, upper=2.0, num_classes=num_classes)
    labels = jnp.arange(12).reshape(2, 3, 2) % num_classes
    class_logits = jax.nn.one_hot(labels, num_classes) * 10.0
    predictions = class_logits.transpose(0, 1, 3, 2).reshape(
        2, 3 * num_classes, 2
    )
    expected = -2.0 + (labels + 0.5) * (4.0 / num_classes)
    distribution = head.get_distribution(predictions)
    for mode in (
        lambda x: head.get_distribution(x).mode(),
        jax.jit(lambda x: head.get_distribution(x).mode()),
    ):
      np.testing.assert_allclose(
          mode(predictions), expected, rtol=1e-6, atol=2e-7
      )
    sample = distribution.sample(seed=jax.random.key(12))
    self.assertEqual(sample.shape, labels.shape)
    self.assertTrue(bool(jnp.isfinite(sample).all()))
    self.assertTrue(bool(((sample >= -2.0) & (sample <= 2.0)).all()))
    value, gradient = jax.value_and_grad(
        lambda p: head.compute_loss(p, expected)
    )(predictions)
    self.assertTrue(bool(jnp.isfinite(value)))
    self.assertTrue(bool(jnp.isfinite(gradient).all()))


if __name__ == '__main__':
  absltest.main()
