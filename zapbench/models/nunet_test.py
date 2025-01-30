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

"""Tests for Nunet models."""

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jnp
import ml_collections

from zapbench.models import nunet


class NunetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = ml_collections.ConfigDict(
        dataclasses.asdict(
            nunet.NunetConfig(
                num_outputs=2,
                embed_dim=8,
                kernel_size=(2, 2, 2),
                num_maxvit_blocks=0,
                num_res_blocks_in=1,
                num_res_blocks_out=1,
                time_conditioning=False,
                remat=True,
                enforce_sharding=False,
                output_type='timesteps',
                combine_type='concat',
            )
        )
    )
    self.rng = jax.random.PRNGKey(42)
    # (batch, t, x, y, z, 1)
    self.x = jax.random.uniform(
        self.rng, (4, 3, 8, 4, 2, 1), minval=0, maxval=1
    )
    self.cond_in = jax.random.uniform(self.rng, (4, 3, 7), minval=0, maxval=1)
    self.cond_out = jax.random.uniform(self.rng, (4, 2, 7), minval=0, maxval=1)
    self.timesteps = jnp.arange(1, 5, dtype=jnp.float32)

  def test_3d_nunet(self):
    model = nunet.Nunet(self.config)
    variables = model.init(self.rng, self.x)
    target_shape = (4, self.config.num_outputs, 8, 4, 2, 1)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_nunet_with_time_conditioning(self):
    self.config.time_conditioning = True
    model = nunet.Nunet(self.config)
    kwargs = {'x': self.x, 'timesteps': self.timesteps}
    variables = model.init(self.rng, **kwargs)
    target_shape = (4, self.config.num_outputs, 8, 4, 2, 1)
    self.assertSequenceEqual(
        target_shape, model.apply(variables, **kwargs).shape
    )

  def test_nunet_feature_output_type(self):
    self.config.output_type = 'features'
    model = nunet.Nunet(self.config)
    variables = model.init(self.rng, self.x)
    target_shape = (4, 1, 8, 4, 2, self.config.num_outputs)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_nunet_with_conditioning(self):
    model = nunet.Nunet(self.config)
    kwargs = {
        'x': self.x,
        'timesteps': self.timesteps,
        'cond_in': self.cond_in,
        'cond_out': self.cond_out,
    }
    variables = model.init(self.rng, **kwargs)
    target_shape = (4, self.config.num_outputs, 8, 4, 2, 1)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_nunet_with_maxvit(self):
    self.config.num_maxvit_blocks = 2
    self.config.maxvit_patch_size = (2, 2, 2)
    self.config.maxvit_num_heads = 2
    model = nunet.Nunet(self.config)
    variables = model.init(self.rng, self.x)
    target_shape = (4, self.config.num_outputs, 8, 4, 2, 1)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_nunet_with_resampling(self):
    self.config.resample_factors = [(1, 1, 1), (4, 2, 1)]
    model = nunet.Nunet(self.config)
    variables = model.init(self.rng, self.x)
    target_shape = (4, self.config.num_outputs, 8, 4, 2, 1)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_nunet_with_concat_upsampling(self):
    self.config.resample_factors = ((1, 1, 1), (4, 2, 1))
    self.config.combine_type = 'concat'
    model = nunet.Nunet(self.config)
    variables = model.init(self.rng, self.x)
    target_shape = (4, self.config.num_outputs, 8, 4, 2, 1)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_nunet_with_upsampling_factors(self):
    self.config.upsample_factors = [(2, 2, 1)]
    model = nunet.Nunet(self.config)
    variables = model.init(self.rng, self.x)
    target_shape = (4, self.config.num_outputs, 8 * 2, 4 * 2, 2, 1)
    self.assertSequenceEqual(target_shape, model.apply(variables, self.x).shape)

  def test_sinusoidal_embeddings(self):
    timesteps = jnp.arange(0, 4, dtype=jnp.float32)
    embeddings = nunet.get_sinusoidal_embedding(timesteps, 32)
    self.assertSequenceEqual(embeddings.shape, (4, 32))
    self.assertNotEqual(embeddings[0].mean(), embeddings[1].mean())


if __name__ == '__main__':
  absltest.main()
