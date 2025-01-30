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

"""Tests for heads."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from zapbench.ts_forecasting import heads


class HeadsTest(absltest.TestCase):

  def test_deterministic_head(self):
    head = heads.DeterministicHead()
    dim_batch, dim_time, dim_features = 2, 3, 4
    predictions = targets = jnp.ones((dim_batch, dim_time, dim_features))
    np.testing.assert_array_equal(
        head.compute_loss(predictions=predictions, targets=targets),
        jnp.zeros((1,)),
    )
    np.testing.assert_array_equal(
        head.get_distribution(predictions).mode(), predictions
    )

  def test_categorical_head(self):
    num_classes = 10
    head = heads.CategoricalHead(num_classes=num_classes)
    dim_batch, dim_time, dim_features = 2, 3, 4
    predictions = jnp.ones((dim_batch, dim_time * num_classes, dim_features))
    targets = jnp.ones((dim_batch, dim_time, dim_features))
    self.assertEqual(
        head.compute_loss(predictions=predictions, targets=targets).shape,
        jnp.array(0.0).shape,
    )
    np.testing.assert_array_equal(
        head.get_distribution(predictions).mode().shape, targets.shape
    )


if __name__ == '__main__':
  absltest.main()
