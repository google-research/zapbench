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

"""Tests for TiDE models."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
from zapbench.models import tide


class TideTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='varying_inputs_1',
          num_batch=12,
          num_timesteps_input=8,
          num_timesteps_output=4,
          num_features=6,
          num_dynamic_covariates=5,
          num_static_covariates=3,
          ablate=False,
          norm=False,
      ),
      dict(
          testcase_name='varying_inputs_2',
          num_batch=6,
          num_timesteps_input=5,
          num_timesteps_output=4,
          num_features=3,
          num_dynamic_covariates=2,
          num_static_covariates=1,
          ablate=False,
          norm=False,
      ),
      dict(
          testcase_name='varying_inputs_3',
          num_batch=1,
          num_timesteps_input=2,
          num_timesteps_output=3,
          num_features=4,
          num_dynamic_covariates=5,
          num_static_covariates=6,
          ablate=False,
          norm=False,
      ),
      dict(
          testcase_name='all_ones_input',
          num_batch=1,
          num_timesteps_input=1,
          num_timesteps_output=1,
          num_features=1,
          num_dynamic_covariates=1,
          num_static_covariates=1,
          ablate=False,
          norm=False,
      ),
      dict(
          testcase_name='ablation',
          num_batch=12,
          num_timesteps_input=8,
          num_timesteps_output=4,
          num_features=6,
          num_dynamic_covariates=5,
          num_static_covariates=3,
          ablate=True,
          norm=False,
      ),
      dict(
          testcase_name='normalization',
          num_batch=12,
          num_timesteps_input=8,
          num_timesteps_output=4,
          num_features=6,
          num_dynamic_covariates=5,
          num_static_covariates=3,
          ablate=False,
          norm=True,
      ),
  )
  def test_Tide(
      self,
      num_batch: int,
      num_timesteps_input: int,
      num_timesteps_output: int,
      num_features: int,
      num_dynamic_covariates: int,
      num_static_covariates: int,
      ablate: bool,
      norm: bool,
  ):
    c = tide.TideConfig(
        pred_len=num_timesteps_output,  # T'
        ablate_past_timeseries=ablate,
        ablate_past_covariates=ablate,
        ablate_future_covariates=ablate,
        ablate_static_covariates=ablate,
        layer_norm=norm,
        instance_norm=norm,
        revert_instance_norm=norm,
    )
    x = jnp.ones((num_batch, num_timesteps_input, num_features))  # BxTxF
    a = jnp.ones((num_features, num_static_covariates))  # FxA
    p = jnp.ones(
        (num_batch, num_timesteps_input, num_dynamic_covariates)
    )  # BxTxC
    f = jnp.ones(
        (num_batch, num_timesteps_output, num_dynamic_covariates)
    )  # BxT'xC
    model = tide.Tide(c)
    key = random.PRNGKey(0)
    params = model.init(key, x, a, p, f, train=False)
    res = model.apply(params, x, a, p, f, train=False)
    self.assertEqual(
        res.shape, (num_batch, num_timesteps_output, num_features)
    )  # BxT'xF


if __name__ == '__main__':
  absltest.main()
