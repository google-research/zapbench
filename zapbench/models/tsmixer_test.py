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

"""Tests for Tsmixer models."""

from absl.testing import absltest
from jax import random
import jax.numpy as jnp
import numpy as np
from zapbench.models import tsmixer


class TsmixerTest(absltest.TestCase):

  def test_Tsmixer(self):
    c = tsmixer.TsmixerConfig(
        pred_len=4,
        instance_norm=False,
    )
    x = jnp.ones((8, 16, 32))
    model = tsmixer.Tsmixer(c)
    key = random.PRNGKey(0)
    params = model.init(key, x, train=False)
    res = model.apply(params, x, train=False)
    self.assertEqual(res.shape, (8, 4, 32))
    for f in range(1, x.shape[-1]):
      with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(res[..., 0], res[..., f])

  def test_Tsmixer_time_mix_only(self):
    c = tsmixer.TsmixerConfig(
        pred_len=4,
        instance_norm=False,
        time_mix_only=True,
    )
    x = jnp.ones((8, 16, 32))
    model = tsmixer.Tsmixer(c)
    key = random.PRNGKey(0)
    params = model.init(key, x, train=False)
    res = model.apply(params, x, train=False)
    self.assertEqual(res.shape, (8, 4, 32))
    for f in range(1, x.shape[-1]):
      np.testing.assert_array_equal(res[..., 0], res[..., f])

if __name__ == '__main__':
  absltest.main()
