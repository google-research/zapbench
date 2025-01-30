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

"""Tests for Nlinear models."""

from absl.testing import absltest
from jax import random
import jax.numpy as jnp
import numpy as np
from zapbench.models import nlinear


class NlinearTest(absltest.TestCase):

  def test_Nlinear(self):
    c = nlinear.NlinearConfig(
        num_outputs=4,
    )
    x = jnp.ones((8, 16, 32))
    model = nlinear.Nlinear(c)
    key = random.PRNGKey(0)
    params = model.init(key, x)
    res = model.apply(params, x)
    self.assertEqual(res.shape, (8, 4, 32))


if __name__ == '__main__':
  absltest.main()
