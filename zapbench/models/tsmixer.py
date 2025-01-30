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

"""Implementation of Tsmixer.

Paper: https://arxiv.org/abs/2303.06053

Reference implementation:
https://github.com/google-research/google-research/tree/master/tsmixer
"""

from typing import Any

from connectomics.jax.models import activation
from connectomics.jax.models import normalization
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp


class TimeMix(nn.Module):
  """Time mixing block for Tsmixer."""

  activation_fn: Any
  norm_layer: Any
  dropout_prob: float
  mlp_dim: int
  residual: bool

  @nn.compact
  def __call__(
      self, x: jax.Array, train: bool) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxTxF in its T-dimension."""
    inputs = x
    x = self.norm_layer()(x.reshape((inputs.shape[0], -1)))  # Bx(T*F)
    x = x.reshape(inputs.shape)  # BxTxF
    x = x.transpose((0, 2, 1))  # BxFxT
    x = nn.Dense(features=self.mlp_dim)(x)
    x = self.activation_fn(x)
    x = x.transpose((0, 2, 1))  # BxTxF
    x = nn.Dropout(self.dropout_prob, deterministic=not train)(x)
    return x + inputs if self.residual else x


class FeatureMix(nn.Module):
  """Feature mixing block for Tsmixer."""

  activation_fn: Any
  norm_layer: Any
  dropout_prob: float
  mlp_dim_1: int
  mlp_dim_2: int
  residual: bool

  @nn.compact
  def __call__(
      self, x: jax.Array, train: bool) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxTxF in its F-dimension."""
    inputs = x
    x = self.norm_layer()(x.reshape((inputs.shape[0], -1)))  # Bx(T*F)
    x = x.reshape(inputs.shape)  # BxTxF
    if self.mlp_dim_1:
      x = nn.Dense(features=self.mlp_dim_1)(x)  # BxTxF'
      x = self.activation_fn(x)
      x = nn.Dropout(self.dropout_prob, deterministic=not train)(x)
    if self.mlp_dim_2:
      x = nn.Dense(features=self.mlp_dim_2)(x)  # BxTxF
      x = nn.Dropout(self.dropout_prob, deterministic=not train)(x)
    return x + inputs if self.residual else x


class MixerBlock(nn.Module):
  """Mixer block for Tsmixer."""

  activation_fn: Any
  norm_layer: Any
  dropout_prob: float
  time_mix_mlp_dim: int
  time_mix_only: bool
  time_mix_residual: bool
  feature_mix_mlp_dim_1: int
  feature_mix_mlp_dim_2: int
  feature_mix_residual: bool
  block_residual: bool

  @nn.compact
  def __call__(self, x: jax.Array, train: bool) ->  jax.Array:
    """Transforms `x` with shape BxTxF -> BxTxF by time and feature mixing."""
    inputs = x

    x = TimeMix(
        activation_fn=self.activation_fn,
        norm_layer=self.norm_layer,
        dropout_prob=self.dropout_prob,
        mlp_dim=self.time_mix_mlp_dim,
        residual=self.time_mix_residual)(x, train=train)
    if self.time_mix_only:
      return x

    x = FeatureMix(
        activation_fn=self.activation_fn,
        norm_layer=self.norm_layer,
        dropout_prob=self.dropout_prob,
        mlp_dim_1=self.feature_mix_mlp_dim_1,
        mlp_dim_2=self.feature_mix_mlp_dim_2,
        residual=self.feature_mix_residual)(x, train=train)

    # Block-level residual connections are not part of original implementation.
    return x + inputs if self.block_residual else x


@struct.dataclass
class TsmixerConfig:
  """Config settings for Tsmixer model."""
  pred_len: int = 1
  instance_norm: bool = True  # At beginning
  revert_instance_norm: bool = True  # At end
  norm: str = ''  # In each block
  activation: str = 'relu'
  n_block: int = 5
  dropout: float = 0.1
  time_mix_only: bool = False
  time_mix_residual: bool = True
  feature_mix_residual: bool = True
  block_residual: bool = False  # Residual connections per block; extra feature
  mlp_dim: int = 100  # If -1, no bottleneck is used
  time_mix_mlp_dim: int = -1  # If -1, use timesteps_input


class Tsmixer(nn.Module):
  """Tsmixer model."""

  config: TsmixerConfig

  @nn.compact
  def __call__(self, x: jax.Array, train: bool) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxT'xF."""

    if self.config.instance_norm:
      rev_in = normalization.ReversibleInstanceNorm()
      x, stats = rev_in(x)

    for i in range(self.config.n_block):
      x = MixerBlock(
          activation_fn=activation.activation_fn_from_str(
              self.config.activation),
          norm_layer=normalization.norm_layer_from_str(self.config.norm, train),
          dropout_prob=self.config.dropout,
          time_mix_mlp_dim=(
              x.shape[-2] if self.config.time_mix_mlp_dim == -1
              else self.config.time_mix_mlp_dim),
          time_mix_only=self.config.time_mix_only,
          time_mix_residual=self.config.time_mix_residual,
          feature_mix_mlp_dim_1=(
              self.config.mlp_dim if self.config.mlp_dim > 0
              else x.shape[-1]),
          feature_mix_mlp_dim_2=(
              x.shape[-1] if self.config.mlp_dim > 0 else 0),
          feature_mix_residual=self.config.feature_mix_residual,
          block_residual=self.config.block_residual,
          name=f'block{i + 1}',
      )(x, train=train)  # BxTxF

    # Temporal projection
    x = x.transpose((0, 2, 1))  # BxFxT
    x = nn.Dense(features=self.config.pred_len)(x)  # BxFxT'
    x = x.transpose((0, 2, 1))  # BxT'xF

    if self.config.instance_norm and self.config.revert_instance_norm:
      x, _ = rev_in(x, stats)  # pylint: disable=undefined-variable

    return x
