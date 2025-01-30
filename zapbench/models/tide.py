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

"""Implementation of TiDE.

Paper: https://arxiv.org/abs/2304.08424

Reference implementation:
https://github.com/google-research/google-research/tree/master/tide
"""

from typing import Callable, Sequence

from connectomics.jax.models import activation
from connectomics.jax.models import normalization
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPResidual(nn.Module):
  """MLPResidual block for Tide."""

  activation_fn: Callable[[jax.Array], jax.Array]
  num_hidden: int
  num_output: int
  dropout_prob: float = 0.0
  layer_norm: bool = False
  use_residual: bool = True

  @nn.compact
  def __call__(self, inputs: jax.Array, train: bool) -> jax.Array:
    x = nn.Dense(
        features=self.num_hidden,
        use_bias=True)(inputs)
    x = self.activation_fn(x)
    x = nn.Dense(
        features=self.num_output,
        use_bias=True)(x)
    x = nn.Dropout(self.dropout_prob, deterministic=not train)(x)
    if self.use_residual:
      x += nn.Dense(
          features=self.num_output,
          use_bias=True)(inputs)
    if self.layer_norm:
      return nn.normalization.LayerNorm()(x)
    return x


class StackedMLPResidual(nn.Module):
  """StackedMLPResidual block for Tide."""

  activation_fn: Callable[[jax.Array], jax.Array]
  num_hiddens: Sequence[int]
  dropout_prob: float = 0.0
  layer_norm: bool = False
  use_residual: bool = True

  @nn.compact
  def __call__(self, inputs: jax.Array, train: bool) -> jax.Array:
    if len(self.num_hiddens) < 2:
      raise ValueError('num_hiddens must be at least contain 2 entries.')
    x = inputs
    for i, num_hidden in enumerate(self.num_hiddens[:-1]):
      x = MLPResidual(
          activation_fn=self.activation_fn,
          num_hidden=num_hidden,
          num_output=self.num_hiddens[i + 1],
          dropout_prob=self.dropout_prob,
          layer_norm=self.layer_norm,
          use_residual=self.use_residual)(x, train=train)
    return x


@struct.dataclass
class TideConfig:
  """Config settings for Tide model."""
  pred_len: int = 1
  past_covariates_num_hidden: int = 128
  past_covariates_dim: int = 32
  future_covariates_num_hidden: int = 128
  future_covariates_dim: int = 32
  encoder_decoder_num_hiddens: Sequence[int] = (128, 128)
  decoder_dim: int = 32
  temporal_decoder_num_hidden: int = 128
  activation: str = 'relu'
  dropout_prob: float = 0.0
  layer_norm: bool = False
  use_residual: bool = True
  instance_norm: bool = False
  revert_instance_norm: bool = False
  ablate_past_timeseries: bool = False
  ablate_static_covariates: bool = False
  ablate_past_covariates: bool = False
  ablate_future_covariates: bool = False


class Tide(nn.Module):
  """Tide model."""

  config: TideConfig

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      static_covariates: jax.Array,
      past_covariates: jax.Array,
      future_covariates: jax.Array,
      train: bool = False,
    ) -> jax.Array:
    """Transforms `x` with shape BxTxF -> BxT'xF using covariates.

    Args:
      x: Input timeseries of shape BxTxF.
      static_covariates: Static covariates of shape FxA.
      past_covariates: Past covariates of shape BxTxC.
      future_covariates: Future covariates of shape BxT'xC.
      train: Whether to run in training mode.

    Returns:
      Output timeseries of shape BxT'xF.
    """
    num_batch, num_timesteps_input, num_features = x.shape  # B, T, F
    num_attributes = static_covariates.shape[1]  # A
    num_covariates = past_covariates.shape[2]  # C
    num_timesteps_output = future_covariates.shape[1]  # T'

    assert static_covariates.shape[0] == num_features
    assert past_covariates.shape[0] == num_batch
    assert past_covariates.shape[1] == num_timesteps_input
    assert future_covariates.shape[0] == num_batch
    assert future_covariates.shape[2] == num_covariates
    assert self.config.pred_len == num_timesteps_output

    if self.config.instance_norm:
      rev_in = normalization.ReversibleInstanceNorm()
      x, stats = rev_in(x)

    # Lookback
    past_timeseries = x.transpose((0, 2, 1))  # BxFxT
    past_timeseries = past_timeseries.reshape(
        num_batch * num_features, num_timesteps_input)  # B*FxT
    if self.config.ablate_past_timeseries:
      past_timeseries = jnp.zeros_like(past_timeseries)

    # Attributes
    static_covariates = jnp.tile(
        static_covariates[jnp.newaxis, ...], (num_batch, 1, 1))  # BxFxA
    static_covariates = static_covariates.reshape(
        num_batch * num_features, num_attributes)  # B*FxA
    if self.config.ablate_static_covariates:
      static_covariates = jnp.zeros_like(static_covariates)

    # Dynamic covariates: Past
    past_covariates = jnp.tile(
        past_covariates[:, jnp.newaxis, :, :], (1, num_features, 1, 1)
    )  # BxFxTxC
    past_covariates = past_covariates.reshape(
        num_batch * num_features, num_timesteps_input, num_covariates
    )  # B*FxTxC
    past_covariates_projected = MLPResidual(
        activation_fn=activation.activation_fn_from_str(
            self.config.activation),
        num_hidden=self.config.past_covariates_num_hidden,
        num_output=self.config.past_covariates_dim,
        dropout_prob=self.config.dropout_prob,
        layer_norm=self.config.layer_norm,
        use_residual=self.config.use_residual
    )(past_covariates, train=train)  # B*FxTxC'
    if self.config.ablate_past_covariates:
      past_covariates_projected = jnp.zeros_like(past_covariates_projected)

    # Dynamic covariates: Future
    future_covariates = jnp.tile(
        future_covariates[:, jnp.newaxis, :, :], (1, num_features, 1, 1)
    )  # B*FxT'xC
    future_covariates = future_covariates.reshape(
        num_batch * num_features, num_timesteps_output, num_covariates
    )  # B*FxT'xC
    future_covariates_projected = MLPResidual(
        activation_fn=activation.activation_fn_from_str(
            self.config.activation),
        num_hidden=self.config.future_covariates_num_hidden,
        num_output=self.config.future_covariates_dim,
        dropout_prob=self.config.dropout_prob,
        layer_norm=self.config.layer_norm,
        use_residual=self.config.use_residual
    )(future_covariates, train=train)  # B*FxT'xC'
    if self.config.ablate_future_covariates:
      future_covariates_projected = jnp.zeros_like(future_covariates_projected)

    # Encoder/Decoder
    x = jnp.concatenate([
        past_timeseries,
        static_covariates,
        past_covariates_projected.reshape(num_batch * num_features, -1),
        future_covariates_projected.reshape(num_batch * num_features, -1),
    ], axis=-1)  # (B*F)x(T+A+C'*T+C'*T')
    x = StackedMLPResidual(
        activation_fn=activation.activation_fn_from_str(
            self.config.activation),
        num_hiddens=self.config.encoder_decoder_num_hiddens,
        dropout_prob=self.config.dropout_prob,
        layer_norm=self.config.layer_norm,
        use_residual=self.config.use_residual)(x, train=train)
    x = StackedMLPResidual(
        activation_fn=activation.activation_fn_from_str(
            self.config.activation),
        num_hiddens=tuple(list(self.config.encoder_decoder_num_hiddens[:-1]) + [
            self.config.pred_len * self.config.decoder_dim,]),
        dropout_prob=self.config.dropout_prob,
        layer_norm=self.config.layer_norm,
        use_residual=self.config.use_residual)(x, train=train)  # B*FxT'*D
    x = x.reshape(num_batch*num_features, self.config.pred_len, -1)  # B*FxT'xD

    # Temporal decoder
    x = jnp.concatenate(
        [x, future_covariates_projected], axis=-1)  # B*FxT'x(D+C')
    x = MLPResidual(
        activation_fn=activation.activation_fn_from_str(
            self.config.activation),
        num_hidden=self.config.temporal_decoder_num_hidden,
        num_output=1,
        dropout_prob=self.config.dropout_prob,
        layer_norm=self.config.layer_norm,
        use_residual=self.config.use_residual
    )(x, train=train)  # B*FxT'x1
    x = x.reshape(num_batch * num_features, self.config.pred_len)  # B*FxT'

    # Residual
    res = nn.Dense(
        features=self.config.pred_len,
        use_bias=True)(past_timeseries)  # B*FxT'
    x += res

    # Reshape
    x = x.reshape(num_batch, num_features, -1)  # BxFxT'
    x = x.transpose((0, 2, 1))  # BxT'xF

    if self.config.instance_norm and self.config.revert_instance_norm:
      x, _ = rev_in(x, stats)  # pylint: disable=undefined-variable
    return x
