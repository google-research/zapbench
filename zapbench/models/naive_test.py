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

"""Tests for naive baseline models."""

from absl.testing import absltest
from jax import random
import jax.numpy as jnp
import numpy as np
from zapbench.models import naive


class NaiveBaselinesTest(absltest.TestCase):

  def test_MeanBaseline(self):
    model = naive.MeanBaseline(naive.MeanBaselineConfig(pred_len=4))
    key = random.PRNGKey(0)
    x = random.normal(key, shape=(8, 4, 32))
    params = model.init(key, x)
    res = model.apply(params, x)
    self.assertEqual(res.shape, (8, 4, 32))
    np.testing.assert_array_equal(
        res, jnp.repeat(x.mean(axis=1, keepdims=True), repeats=4, axis=1))

  def test_ReturnCovariates(self):
    x = jnp.ones((8, 16, 32))
    model = naive.ReturnCovariates(naive.ReturnCovariatesConfig())
    key = random.PRNGKey(0)
    cov = random.normal(key, shape=(8, 4, 32))
    params = model.init(key, x, cov)
    res = model.apply(params, x, cov)
    np.testing.assert_array_equal(res, cov)

if __name__ == '__main__':
  absltest.main()
