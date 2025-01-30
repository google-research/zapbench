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

"""Implementation of Nlinear.

Normalization is optional.

Paper: https://arxiv.org/abs/2205.13504

Reference implementation:
https://github.com/cure-lab/LTSF-Linear
"""

from connectomics.jax.models import initializer
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp


@struct.dataclass
class NlinearConfig:
  """Config settings for Nlinear model.

  Attributes:
    num_outputs: Number of outputs.
    constant_init: If True, initialize weights for equal weighting of inputs.
    normalization: Whether to apply normalization, i.e., subtract the last
      timestep before applying the linear layer and add it back afterwards.
  """
  num_outputs: int = 1
  constant_init: bool = True
  normalization: bool = True


class Nlinear(nn.Module):
  """Nlinear model."""

  config: NlinearConfig

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxT'xF."""
    del train  # Unused by model; included for consistent signature.
    if self.config.normalization:
      last_step = x[:, -1:, :]
      x = x - last_step

    x = x.transpose((0, 2, 1))  # BxFxT
    x = nn.Dense(
        features=self.config.num_outputs,
        kernel_init=(initializer.constant_init(dim=0)
                     if self.config.constant_init
                     else nn.initializers.lecun_normal()),
        use_bias=True)(x)  # BxFxT'
    x = x.transpose((0, 2, 1))  # BxT'xF

    if self.config.normalization:
      return x + last_step  # pylint: disable=undefined-variable
    else:
      return x
