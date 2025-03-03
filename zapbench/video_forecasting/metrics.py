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

"""Metrics for volumetric video forecasting.

All metrics assume a leading batch dimension that is preserved and assume
inputs `predictions`, `targets`. All functions are compatible with clu.metrics
constructed from functions. Constructed relative metrics additionally require
the keyword argument `baseline`.
"""

from collections.abc import Callable
import functools as ft

import chex
# pylint: disable=g-importing-member, unused-import
from connectomics.jax.metrics import mae
from connectomics.jax.metrics import make_per_step_metric
from connectomics.jax.metrics import make_relative_metric
from connectomics.jax.metrics import mape
from connectomics.jax.metrics import mse
from connectomics.jax.metrics import PerStepAverage
# pylint: enable=g-importing-member, unused-import
import jax
import jax.numpy as jnp
import numpy as np


def _segment_mean(
    data: jnp.ndarray, segment_ids: jnp.ndarray, elems_per_segment: jnp.ndarray
) -> jnp.ndarray:
  num_segments = elems_per_segment.shape[0]
  segment_sums = jax.ops.segment_sum(data, segment_ids, num_segments)
  return segment_sums / elems_per_segment


def _segment_mean_non_background(
    data: jnp.ndarray, segment_ids: jnp.ndarray, elems_per_segment: jnp.ndarray
) -> jnp.ndarray:
  """Compute mean over segments ignoring index zero as background."""
  return _segment_mean(data, segment_ids, elems_per_segment)[1:]


def extract_traces(
    voxels: jnp.ndarray, trace_mask: jnp.ndarray, trace_counts: jnp.ndarray
) -> jnp.ndarray:
  """Extract traces from voxels based on given mask."""
  rank_vol, rank_mask = voxels.ndim, trace_mask.ndim
  assert rank_vol >= rank_mask, 'Mask has to be have at least rank of volume.'
  leading_shape = voxels.shape[: rank_vol - rank_mask]
  leading_elems = int(np.prod(leading_shape))
  seg_mean_vmap = jax.vmap(
      ft.partial(
          _segment_mean_non_background,
          segment_ids=trace_mask.reshape(-1),
          elems_per_segment=trace_counts,
      )
  )
  return seg_mean_vmap(voxels.reshape(leading_elems, -1)).reshape(
      *leading_shape, -1
  )


def _nan_to_zero_traces(
    trace_predictions: jnp.ndarray, trace_targets: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Convert nans to zeros in trace predictions and targets."""
  kwargs = dict(nan=0.0, posinf=0.0, neginf=0.0)
  trace_predictions = jnp.nan_to_num(trace_predictions, **kwargs)
  trace_targets = jnp.nan_to_num(trace_targets, **kwargs)
  return trace_predictions, trace_targets


def make_trace_based_metric(
    metric: Callable[..., jnp.ndarray],
) -> Callable[..., jnp.ndarray]:
  """Construct trace-based metric with mask and voxel counts."""

  def _trace_metric(
      predictions: jnp.ndarray,
      targets: jnp.ndarray,
      trace_mask: jnp.ndarray,
      trace_counts: jnp.ndarray,
      has_channel: bool = True,
      nan_to_zero: bool = False,
      **kwargs,
  ) -> jnp.ndarray:
    if has_channel:  # add channel to mask
      trace_mask = trace_mask.reshape((*trace_mask.shape, -1))
    assert predictions.shape == targets.shape
    trace_predictions = extract_traces(predictions, trace_mask, trace_counts)
    trace_targets = extract_traces(targets, trace_mask, trace_counts)
    if nan_to_zero:
      trace_predictions, trace_targets = _nan_to_zero_traces(
          trace_predictions, trace_targets
      )
    return metric(
        predictions=trace_predictions, targets=trace_targets, **kwargs
    )

  return _trace_metric


def make_trace_based_metric_with_extracted_traces(
    metric: Callable[..., jnp.ndarray],
) -> Callable[..., jnp.ndarray]:
  """Construct trace-based metric based with extracted traces."""

  def _trace_metric(
      trace_predictions: jnp.ndarray,
      trace_targets: jnp.ndarray,
      nan_to_zero: bool = False,
      **kwargs,
  ) -> jnp.ndarray:
    if nan_to_zero:
      trace_predictions, trace_targets = _nan_to_zero_traces(
          trace_predictions, trace_targets
      )
    return metric(
        predictions=trace_predictions,
        targets=trace_targets,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ('predictions', 'targets')
        },
    )

  return _trace_metric


def psnr(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    dynamic_range: float = 1.0,
    video: bool = False,
    **_,
) -> jnp.ndarray:
  """Compute peak signal-to-noise ratio per example."""
  assert predictions.shape == targets.shape
  log_dr = jnp.log10(dynamic_range)
  axes = tuple(range(2 if video else 1, targets.ndim))
  mses = jnp.mean(jnp.square(targets - predictions), axis=axes)
  log10_mses = jnp.log10(mses).mean(axis=1) if video else jnp.log10(mses)
  return 20.0 * log_dr - 10.0 * log10_mses


def ssim(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    dynamic_range: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    dim: int = 2,
    video: bool = False,
    has_channel: bool = True,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    **_,
) -> chex.Numeric:
  """Computes the structural similarity index (SSIM) between volume/image pairs.

  Adapted from dm_pix.metrics.py and extended to volumes and videos. dm_pix
  only handles batches of images. This function instead flexibly handles the
  following cases where an asterisk indicates an optional dimension:
  (batch*, frames*, z*, x, y, channels)
  Frames and spatial dimensions are handled differently according to SSIM, i.e.,
  each frame is treated separately and not convolved/filtered over. Argument
  dim enables z with dim=3 and disables it with dim=2, boolean argument video
  toggles video axis. These have to be set correctly to infer shapes.

  This function is based on the standard SSIM implementation from:
  Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
  "Image quality assessment: from error visibility to structural similarity",
  in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

  Args:
    predictions: images or volumes (with or without batch dimension).
    targets: reference of the same shape as `predictions`.
    dynamic_range: The maximum magnitude that `a` or `b` can have.
    filter_size: Window size (>= 1). Image dims must be at least this small.
    filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
    k1: One of the SSIM dampening parameters (> 0.).
    k2: One of the SSIM dampening parameters (> 0.).
    return_map: If True, will cause the per-pixel SSIM "map" to be returned.
    dim: dimensionality (2 -> img, 3 -> volume).
    video: whether data is video -> average ssim over frames.
    has_channel: whether data has channel (required for shape checks)
    precision: The numerical precision to use when performing convolution.

  Returns:
    Each volume's mean SSIM, or a tensor of individual values if `return_map`.
  """
  chex.assert_type([predictions, targets], float)
  chex.assert_equal_shape([predictions, targets])
  assert dim in (2, 3)
  if not has_channel:
    predictions = predictions.reshape((*predictions.shape, 1))
    targets = targets.reshape((*targets.shape, 1))
  rank = len(targets.shape)
  exp_rank = (dim + 1) + (1 if video else 0)
  # with or without batch dimension
  chex.assert_rank(predictions, {exp_rank, exp_rank + 1})
  has_batch_axis = rank == exp_rank + 1
  a, b = predictions, targets
  if video and has_batch_axis:  # fold video into batch axis
    batch_dim = a.shape[0]
    a = a.reshape((-1, *a.shape[2:]))
    b = b.reshape((-1, *b.shape[2:]))
  else:
    batch_dim = None

  # Construct a 1D Gaussian blur filter.
  hw = filter_size // 2
  shift = (2 * hw - filter_size + 1) / 2
  f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
  filt = jnp.exp(-0.5 * f_i)
  filt /= jnp.sum(filt)

  # Construct a 1D convolution.
  filt_fn_1 = lambda z: jnp.convolve(z, filt, mode='valid', precision=precision)
  filt_fn_vmap = jax.vmap(filt_fn_1)

  def _filt_fn(v, axis) -> chex.Array:
    v_flat = jnp.moveaxis(v, axis, -1).reshape((-1, v.shape[axis]))
    v_filt_shape = (v.shape[0],) if has_batch_axis else ()
    if dim == 3 and axis != -4:
      v_filt_shape += (v.shape[-4],)
    if axis != -3:
      v_filt_shape += (v.shape[-3],)
    if axis != -2:
      v_filt_shape += (v.shape[-2],)
    if axis != -1:
      v_filt_shape += (v.shape[-1],)
    v_filt_shape += (-1,)
    return jnp.moveaxis(filt_fn_vmap(v_flat).reshape(v_filt_shape), -1, axis)

  # Apply the blur in both x and y (and z for volumes).
  # TODO(aleximmer): consider direct nd filter and benchmark speed.
  if dim == 3:
    filt_fn = lambda z: _filt_fn(_filt_fn(_filt_fn(z, -2), -3), -4)
  else:
    filt_fn = lambda z: _filt_fn(_filt_fn(z, -2), -3)

  mu0 = filt_fn(a)
  mu1 = filt_fn(b)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filt_fn(a**2) - mu00
  sigma11 = filt_fn(b**2) - mu11
  sigma01 = filt_fn(a * b) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  epsilon = jnp.finfo(jnp.float32).eps ** 2
  sigma00 = jnp.maximum(epsilon, sigma00)
  sigma11 = jnp.maximum(epsilon, sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
  )

  c1 = (k1 * dynamic_range) ** 2
  c2 = (k2 * dynamic_range) ** 2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  if video and has_batch_axis:  # unfold video axis and optionally avg. over it
    ssim_map = ssim_map.reshape((batch_dim, -1, *ssim_map.shape[1:]))
  ssim_value = jnp.mean(ssim_map, list(range(1 if has_batch_axis else 0, rank)))
  return ssim_map if return_map else ssim_value
