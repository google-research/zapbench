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

"""Losses."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jax.scipy.special


@dataclasses.dataclass(frozen=True, kw_only=True)
class GaussianHistogramLoss:
  """Histogram loss transform for a normal distribution.

  Based on Listing 1 in Farebrother et al., 2024.
  """

  num_bins: int
  sigma_ratio: float
  min_value: float
  max_value: float

  @property
  def bin_width(self) -> float:
    return (self.max_value - self.min_value) / self.num_bins

  @property
  def sigma(self) -> float:
    return self.bin_width * self.sigma_ratio

  @functools.cached_property
  def support(self) -> jax.Array:
    return jnp.linspace(
        self.min_value, self.max_value, self.num_bins + 1, dtype=jnp.float32
    )

  def transform_to_probs(self, target: jax.Array) -> jax.Array:
    cdf_evals = jax.scipy.special.erf(
        (self.support - target) / (jnp.sqrt(2) * self.sigma)
    )
    z = cdf_evals[-1] - cdf_evals[0]
    bin_probs = cdf_evals[1:] - cdf_evals[:-1]
    return bin_probs / z

  def transform_from_probs(self, probs: jax.Array) -> jax.Array:
    centers = (self.support[:-1] + self.support[1:]) / 2
    return jnp.sum(probs * centers)
