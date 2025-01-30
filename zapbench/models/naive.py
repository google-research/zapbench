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

"""Implementation of naive baseline models."""

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp


@struct.dataclass
class MeanBaselineConfig:
  """Config settings for mean baseline model."""
  pred_len: int = 1


class MeanBaseline(nn.Module):
  """Mean baseline model."""

  config: MeanBaselineConfig

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxT'xF."""
    del train  # Unused by model; included for consistent signature.
    # Unused parameter s.t. model can be used with existing code that does not
    # account for parameter-free models.
    self.param('unused_param', nn.initializers.zeros, (1,))
    return jnp.repeat(
        x.mean(axis=1, keepdims=True), repeats=self.config.pred_len, axis=1)


@struct.dataclass
class ReturnCovariatesConfig:
  """Config settings for model that returns covariates."""


class ReturnCovariates(nn.Module):
  """Model that returns covariates."""

  config: ReturnCovariatesConfig  # Unused; included for consistent signature.

  @nn.compact
  def __call__(
      self, x: jax.Array, cov: jax.Array, train: bool = False) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxT'xF."""
    del train, x  # Unused by model; included for consistent signature.
    # Unused parameter s.t. model can be used with existing code that does not
    # account for parameter-free models.
    self.param('unused_param', nn.initializers.zeros, (1,))
    return cov
