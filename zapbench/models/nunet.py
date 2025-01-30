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

"""Normalized UNet-like models.

Use input dimensions as features rather than channels.
"""

from collections.abc import Callable
import functools as ft
import math
from typing import Sequence

from absl import logging
import chex
from connectomics.jax.models import attention as attn
from flax import struct
import flax.linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp


DENSE_INIT = initializers.variance_scaling(
    scale=1.0 / 3, mode='fan_out', distribution='uniform'
)


@struct.dataclass
class NunetConfig:
  """Config settings for video convstack applied to 2-3D + T inputs.

  Nunet starts with a downsampling stack specified by resample_factors using
  embed_dim features and an optional MaxViT at the lowest resolution.
  Each resampling block is followed by a stack of residual blocks, and uses
  a unet upsampling stack. Lastly, an optional upsampling stack is applied
  for a superresolution output, which uses upsample_dim features. The output
  is then projected to num_outputs features, which are timesteps if
  output_type == 'timesteps' and features otherwise.

  Attributes:
    num_outputs: number of output timesteps or features
    output_type: whether to use output as features (last dim) or timesteps (1st)
    embed_dim: number of base feature dimensions
    kernel_size: size of kernel for (x, y, (z,))
    num_res_blocks_in: number of input residual blocks per resolution (each with
      2 convolutions).
    num_res_blocks_out: number of output residual blocks per resolution (each
      with 2 convolutions).
    num_res_blocks_up: number of residual blocks in the upsampling stack (each
      with 2 convolutions).
    resample_factors: factors by which to resample the input at each resolution.
    upsample_factors: factors by which to upsample the output.
    upsample_dim: feature dimension of upsampling layer.
    combine_type: how to combine the resampled inputs (add or concat).
    separate_z: whether to separate z from (x, y) for convolutions.
    num_maxvit_blocks: number of consecutive multi-axis vision transformer
      blocks at the lowest resolution for global receptive field.
    maxvit_num_heads: number of heads in multi-axis multi-head attention.
    maxvit_patch_size: patch sizes per spatial dimension used in MaxViT, for
      example 3-dimensional when using full (x, y, z)-shaped input.
    norm: type of normalization layer, e.g., 'layernorm', 'batchnorm', or
      'groupnorm_{num_grups}'
    activation: activation function to use
    channel_dropout: dropout for channel dimension (timesteps) in resnet blocks
    spatial_dropout: dropout for spatial dimensions (x, y, (z,)) at input
    attention_dropout: dropout for attention layers
    conditioning: whether to use conditioning variables passed with film layer
    time_conditioning: whether to condition on lead time (instead of using auto-
      regressive or direct forecasting)
    remat: whether to remat resnet blocks to potentially save memory.
    enforce_sharding: whether to enforce sharding with constraints.
  """

  num_outputs: int = 1
  output_type: str = 'timesteps'
  embed_dim: int = 64
  kernel_size: tuple[int, ...] = (3, 3, 3)
  num_res_blocks_in: int | tuple[int, ...] = 0
  num_res_blocks_out: int | tuple[int, ...] = 0
  num_res_blocks_up: int = 0
  resample_factors: Sequence[tuple[int, ...]] = ()
  upsample_factors: Sequence[tuple[int, ...]] = ()
  upsample_dim: int = 16
  combine_type: str = 'add'
  separate_z: bool = False
  num_maxvit_blocks: int = 0
  maxvit_num_heads: int = 32
  maxvit_patch_size: tuple[int, ...] = (8, 8, 9)
  norm: str = 'layernorm'
  activation: str = 'swish'
  channel_dropout: float = 0.0
  spatial_dropout: float = 0.0
  attention_dropout: float = 0.0
  conditioning: bool = False
  time_conditioning: bool = False
  remat: bool = True
  enforce_sharding: bool = True


def get_activation(name: str) -> Callable[..., jax.Array]:
  linear = lambda x: x
  return {
      'swish': nn.swish,
      'relu': nn.relu,
      'linear': linear,
      'gelu': nn.gelu,
  }[name]


def get_normalization(
    name: str, train: bool, layer_name: str
) -> Callable[..., jax.Array]:
  """Obtain normalization layer from configuration strings.

  Args:
    name: normalization type with potential hyperparams (e.g. for groupnorm)
    train: whether in train mode
    layer_name: name to give the layer

  Returns:
    normalization function
  """
  if name == 'none':
    return lambda x: x
  elif 'groupnorm' in name:
    num_groups = int(name.split('_')[1])
    return nn.normalization.GroupNorm(num_groups=num_groups, name=layer_name)
  elif name == 'layernorm':
    return nn.normalization.LayerNorm(name=layer_name)
  elif name == 'batchnorm':
    return nn.normalization.BatchNorm(
        use_running_average=not train, name=layer_name
    )
  else:
    raise ValueError(f'Invalid normalization {name}.')


def get_sinusoidal_embedding(timesteps: jax.Array, dim: int) -> jax.Array:
  # Sinusoidal embeddings as proposed in "Attention is all you need" paper.
  divs = jnp.exp(
      -math.log(1e4) * jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
  )
  arguments = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(divs, 0)
  return jnp.concatenate([jnp.cos(arguments), jnp.sin(arguments)], axis=1)


def condition(
    x: jax.Array, cond: jax.Array | None, suffix: str = ''
) -> jax.Array:
  """Conditions input x on conditioning variable cond unless cond is None."""
  if cond is None:
    return x
  cond_film = nn.Dense(
      features=x.shape[-1] * 2,
      kernel_init=nn.initializers.zeros,
      name=f'cond_{suffix}',
  )(cond)
  cond_film = jnp.expand_dims(cond_film, axis=range(1, x.ndim - 1))
  shift, scale = jnp.split(cond_film, 2, axis=-1)
  return shift + x * (scale + 1)


def downsample(x: jax.Array, factor: tuple[int, ...]) -> jax.Array:
  return nn.avg_pool(x, window_shape=factor, strides=factor, padding='same')


def upsample(x: jax.Array, factor: tuple[int, ...]) -> jax.Array:
  batch = x.shape[0]
  features = x.shape[-1]
  spatial_out_shape = tuple(e * f for e, f in zip(x.shape[1:-1], factor))
  shape = (batch, *(spatial_out_shape), features)
  return jax.image.resize(x, shape=shape, method='nearest')


def swap_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
  return shape[-1:] + shape[:-1]


class DenseBlock(nn.Module):
  """Dense block."""

  num_layers: int
  num_features: int
  activation: str
  linear_out: bool = False

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    for layer in range(self.num_layers):
      x = nn.Dense(
          features=self.num_features,
          name=f'dense_{layer}',
          kernel_init=DENSE_INIT,
      )(x)
      if layer < self.num_layers - 1 or not self.linear_out:
        x = get_activation(self.activation)(x)
    return x


class ResBlock(nn.Module):
  """Pre-activation convolutional residual block."""

  norm: str
  activation: str
  kernel_size: tuple[int, ...]
  features: int
  dropout: float
  train: bool

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      cond_in: jax.Array | None,
      cond_out: jax.Array,
      cond_time: jax.Array | None,
  ) -> jax.Array:
    norm = ft.partial(get_normalization, self.norm, self.train)
    act = get_activation(self.activation)
    conv_kwargs = dict(features=self.features, kernel_size=self.kernel_size)

    h = norm('norm_1')(x)
    h = act(h)
    h = nn.Conv(**conv_kwargs, name='conv_1')(h)
    h = norm('norm_2')(h)
    h = condition(h, cond_in, 'in')
    h = condition(h, cond_out, 'out')
    h = condition(h, cond_time, 'time')
    h = act(h)
    h = nn.Dropout(
        rate=self.dropout,
        broadcast_dims=range(1, h.ndim - 1),
    )(h, deterministic=not self.train)
    h = nn.Conv(**conv_kwargs, name='conv_2')(h)
    if x.shape[-1] != h.shape[-1]:
      x = nn.Dense(features=h.shape[-1], name='shortcut')(x)
    return x + h


class MaxVitBlock(nn.Module):
  """Multi-axis vision transformer block (block & grid attention, MLPs)."""

  patch_size: tuple[int, ...]
  num_heads: int
  dropout: float
  x_shard_fn: Callable[[jax.Array], jax.Array]
  train: bool

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      cond_in: jax.Array | None,
      cond_out: jax.Array | None,
      cond_time: jax.Array | None,
  ) -> jax.Array:
    dense_kwargs = dict(
        num_layers=2,
        num_features=x.shape[-1],
        activation='gelu',
        linear_out=True,
    )
    attn_kwargs = dict(
        num_heads=self.num_heads,
        qkv_features=None,
        dropout=self.dropout,
        positional_embed=True,
        relative_attention_bias=True,
        patch_sizes=self.patch_size,
        seq_shard_fn=self.x_shard_fn,
    )

    h = nn.LayerNorm(name='norm_block')(x)
    h = condition(h, cond_in, 'in_block')
    h = condition(h, cond_out, 'out_block')
    h = condition(h, cond_time, 'time_block')
    h = attn.BlockAttention(**attn_kwargs, name='attn_block')(h, self.train)
    x = x + h
    h = nn.LayerNorm(name='norm_mlp_block')(x)
    h = DenseBlock(**dense_kwargs, name='mlp_block')(h)
    x = x + h

    h = nn.LayerNorm(name='norm_grid')(x)
    h = condition(h, cond_in, 'in_grid')
    h = condition(h, cond_out, 'out_grid')
    h = condition(h, cond_time, 'time_grid')
    h = attn.GridAttention(**attn_kwargs, name='attn_grid')(h, self.train)
    x = x + h
    h = nn.LayerNorm(name='norm_mlp_grid')(x)
    h = DenseBlock(**dense_kwargs, name='mlp_grid')(h)
    return x + h


class Nunet(nn.Module):
  """Normalized UNet with 2-3D + T inputs."""

  config: NunetConfig

  @nn.compact
  def __call__(
      self,
      x: chex.Array,
      cond_in: chex.Array | None = None,
      cond_out: chex.Array | None = None,
      timesteps: chex.Array | None = None,
      train: bool = True,
      x_sharding: jax.sharding.Sharding | None = None,
      **kwargs,
  ) -> chex.Array:
    """Applies normalized UNet to input data x and optional lead timesteps.

    Args:
      x: [batch, t, x, y, z, 1]-shaped input
      cond_in: [batch, t, f]-shaped input condition
      cond_out: [batch, t, f]-shaped output condition
      timesteps: [batch,] lead time to forecast at
      train: whether in train mode
      x_sharding: how input x is sharded
      **kwargs: keyword arguments for interface compatibility

    Returns:
      Nunet output.
    """
    if x.ndim != 6:
      raise ValueError(
          f'Expected input of shape [batch, t, x, y, z, 1], got {x.shape}'
      )
    # turn time into channel dimension to have [batch, z, x, y, t]
    x = jnp.squeeze(x, axis=-1)
    x = jnp.swapaxes(x, 1, -1)
    logging.info('Input shape %r', x.shape)
    kernel_size = swap_axes(self.config.kernel_size)
    maxvit_patch_size = swap_axes(self.config.maxvit_patch_size)
    resample_factors = [swap_axes(f) for f in self.config.resample_factors]
    upsample_factors = [swap_axes(f) for f in self.config.upsample_factors]
    resolutions = len(resample_factors) + 1
    if isinstance(self.config.num_res_blocks_in, int):
      num_res_blocks_in = resolutions * (self.config.num_res_blocks_in,)
    else:
      num_res_blocks_in = self.config.num_res_blocks_in
    if isinstance(self.config.num_res_blocks_out, int):
      num_res_blocks_out = resolutions * (self.config.num_res_blocks_out,)
    else:
      num_res_blocks_out = self.config.num_res_blocks_out

    remat_fn = (
        nn.remat
        if self.config.remat and not self.is_initializing()
        else lambda f: f
    )
    x_shard_fn = (
        ft.partial(jax.lax.with_sharding_constraint, shardings=x_sharding)
        if self.config.enforce_sharding and not self.is_initializing()
        else lambda x: x
    )

    # embed the conditioning stimulus and time information using a dense block
    dense_kwargs = dict(
        num_layers=1,
        num_features=256,
        activation='swish',
        linear_out=False,
    )

    if self.config.conditioning:
      if cond_in is None or cond_out is None:
        raise ValueError('Conditioning requires inputs cond_in and cond_out')
      cond_in = cond_in.reshape(cond_in.shape[0], -1)  # [batch, t_in * f]
      cond_in = DenseBlock(**dense_kwargs, name='cond_in_embed')(cond_in)
      cond_out = cond_out.reshape(cond_out.shape[0], -1)  # [batch, t_out * f]
      cond_out = DenseBlock(**dense_kwargs, name='cond_out_embed')(cond_out)
      logging.info('Emb. cond_in %r cond_out %r', cond_in.shape, cond_out.shape)
    else:
      cond_in, cond_out = None, None
      logging.info('No conditioning')

    if self.config.time_conditioning:
      cond_time = get_sinusoidal_embedding(timesteps, 32)
    else:
      cond_time = None

    conv_kwargs = dict(
        features=self.config.embed_dim,
        kernel_size=kernel_size,
    )
    conv = remat_fn(nn.Conv)
    res_block_kwargs = dict(
        norm=self.config.norm,
        activation=self.config.activation,
        kernel_size=kernel_size,
        features=self.config.embed_dim,
        dropout=self.config.channel_dropout,
        train=train,
    )
    norm = ft.partial(get_normalization, self.config.norm, train)
    act = get_activation(self.config.activation)

    # spatial dropout to promote learning multivariate patterns
    x = nn.Dropout(
        rate=self.config.spatial_dropout, broadcast_dims=(0, x.ndim - 1)
    )(x, deterministic=not train)

    # embed the input volume
    x = conv(**conv_kwargs, name='conv_in')(x)
    x = x_shard_fn(x)
    for i in range(num_res_blocks_in[0]):
      x = remat_fn(ResBlock)(**res_block_kwargs, name=f'res_in0_b{i}')(
          x, cond_in, cond_out, cond_time
      )
      x = x_shard_fn(x)
    xs_resampled = [x]

    # create downsampled versions of the input and apply resnet blocks
    num_res_blocks = num_res_blocks_in[1:]
    for d, (factor, blocks) in enumerate(zip(resample_factors, num_res_blocks)):
      x_res = x_shard_fn(downsample(xs_resampled[-1], factor))
      logging.info('Downsampled to shape %r', x_res.shape)
      r = d + 1
      for i in range(blocks):
        x_res = remat_fn(ResBlock)(**res_block_kwargs, name=f'res_in{r}_b{i}')(
            x_res, cond_in, cond_out, cond_time
        )
        x_res = x_shard_fn(x_res)
      xs_resampled.append(x_res)

    # feature learning and upsampling stack
    for d, (x_resampled, resample_factor, blocks) in enumerate(
        zip(
            reversed(xs_resampled),
            reversed([None] + resample_factors),
            reversed(num_res_blocks_out),
        )
    ):
      if d > 0:  # connect residual from downsampling
        if self.config.combine_type == 'add':
          x = x_resampled + x
        elif self.config.combine_type == 'concat':
          x = jnp.concatenate([x_resampled, x], axis=-1)
        else:
          raise ValueError(
              f'Unsupported combine type: {self.config.combine_type}'
          )
        x = x_shard_fn(x)
      else:
        x = x_resampled

      # MaxViT at lowest resolution for global receptive field
      if d == 0:
        for i in range(self.config.num_maxvit_blocks):
          x = remat_fn(MaxVitBlock)(
              patch_size=maxvit_patch_size,
              num_heads=self.config.maxvit_num_heads,
              dropout=self.config.attention_dropout,
              x_shard_fn=x_shard_fn,
              name=f'maxvit_{i}',
              train=train,
          )(x, cond_in, cond_out, cond_time)
          x = x_shard_fn(x)

      for i in range(blocks):
        x = remat_fn(ResBlock)(**res_block_kwargs, name=f'res_out{d}_b{i}')(
            x, cond_in, cond_out, cond_time
        )
        x = x_shard_fn(x)

      if d < len(resample_factors):
        x = upsample(x, resample_factor)
        logging.info('Upsampled to shape %r', x.shape)

    # Upsample to superresolution
    conv_kwargs['features'] = self.config.upsample_dim
    res_block_kwargs['features'] = self.config.upsample_dim
    if upsample_factors:
      # project to different embedding dimension first
      x = norm('norm_re')(x)
      x = act(x)
      x = conv(**conv_kwargs, name='conv_re')(x)
      x = x_shard_fn(x)

    for u, upsample_factor in enumerate(upsample_factors):
      x = upsample(x, upsample_factor)
      logging.info('Upsampled to shape %r', x.shape)
      x = x_shard_fn(x)

      for i in range(self.config.num_res_blocks_up):
        x = remat_fn(ResBlock)(**res_block_kwargs, name=f'res_up{u}_b{i}')(
            x, cond_in, cond_out, cond_time
        )
        x = x_shard_fn(x)

    # project to output
    x = norm('norm_out')(x)
    x = condition(x, cond_in, 'in_out')
    x = condition(x, cond_out, 'out_out')
    x = condition(x, cond_time, 'time_out')
    x = act(x)
    x = nn.Dropout(
        rate=self.config.channel_dropout, broadcast_dims=range(1, x.ndim - 1)
    )(x, deterministic=not train)
    conv_kwargs['features'] = self.config.num_outputs
    x = conv(**conv_kwargs, name='conv_out')(x)
    x = x_shard_fn(x)

    # move z back to the end of spatial dimensions
    x = jnp.moveaxis(x, 1, -2)
    logging.info('Output shape %r', x.shape)
    # we have [batch, x, y, z, num_outputs]
    # Add squeezed dim to have [batch, 1, x, y, z, num_outputs]
    x = jnp.expand_dims(x, axis=1)
    if self.config.output_type == 'features':
      return x  # (batch, 1, x, y, z, num_outputs)
    elif self.config.output_type == 'timesteps':
      return jnp.swapaxes(x, -1, 1)  # (batch, num_outputs, x, y, z, 1)
    else:
      raise ValueError(f'Unknown output type: {self.config.output_type}')
