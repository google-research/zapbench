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

"""Tests for metrics.

Mostly tested against dm_pix and skimage libraries. skimage does not permit
acceleration using jax and dm_pix is limited to [batch, x, y, channel].
"""

from absl.testing import absltest
from absl.testing import parameterized
import dm_pix
import numpy as np
import skimage

from zapbench.video_forecasting import metrics



class MetricsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(42)
    # (batch, z, y, x, channel)
    self.vol1 = rng.uniform(0, 1, (4, 28, 32, 36, 2))
    self.vol2 = rng.uniform(0, 1, (4, 28, 32, 36, 2))
    # skimage kwargs to get behaviour of Wang et al. (2004), the default we
    # and dm_pix use.
    self.sk_kwargs = dict(K1=0.01, K2=0.03, win_size=11, data_range=1.0,
                          channel_axis=-1, sigma=1.5, gaussian_weights=True,
                          use_sample_covariance=False)

  def test_psnr_2d(self):
    psnr_pix = dm_pix.psnr(self.vol1[:, 0], self.vol2[:, 0])
    psnr_neuro = metrics.psnr(self.vol1[:, 0], self.vol2[:, 0], video=False)
    np.testing.assert_allclose(psnr_pix, psnr_neuro, atol=1e-6, rtol=1e-6)

  def test_psnr_video(self):
    # for video, we compute the average psnr over all frames
    psnr_pix = dm_pix.psnr(self.vol1[0, :], self.vol2[0, :]).mean()
    psnr_neuro = metrics.psnr(self.vol1[0:1, :], self.vol2[0:1, :], video=True)
    np.testing.assert_allclose(psnr_pix, psnr_neuro, atol=1e-6, rtol=1e-6)

  def test_ssim_integration_against_pix_2d(self):
    # dm_pix only supports 2D
    ssim_pix = dm_pix.ssim(self.vol1[:, 0], self.vol2[:, 0])
    ssim_neuro = metrics.ssim(self.vol1[:, 0], self.vol2[:, 0], dim=2)
    self.assertSequenceAlmostEqual(ssim_pix, ssim_neuro)

  def test_ssim_integration_against_skimage_2d(self):
    # skimage cannot deal with batch-dimension
    x = self.vol1[0, 0]
    y = self.vol2[0, 0]
    ssim_skimage = skimage.metrics.structural_similarity(x, y, **self.sk_kwargs)
    ssim_jax = metrics.ssim(x, y, dim=2)
    np.testing.assert_allclose(ssim_jax, ssim_skimage, atol=1e-6, rtol=1e-6)

  def test_ssim_integration_against_skimage_3d(self):
    # skimage cannot deal with batch
    x = self.vol1[0]
    y = self.vol2[0]
    ssim_skimage = skimage.metrics.structural_similarity(x, y, **self.sk_kwargs)
    ssim_jax = metrics.ssim(x, y, dim=3)
    np.testing.assert_allclose(ssim_jax, ssim_skimage, atol=1e-6)

  def test_ssim_batching(self):
    # 2D
    ssim_single = metrics.ssim(self.vol1[0, 0], self.vol2[0, 0], dim=2)
    ssim_batched = metrics.ssim(self.vol1[:, 0], self.vol2[:, 0], dim=2)
    self.assertEqual(ssim_single, ssim_batched[0])
    # 3D
    ssim_single = metrics.ssim(self.vol1[0], self.vol2[0], dim=3)
    ssim_batched = metrics.ssim(self.vol1, self.vol2, dim=3)
    self.assertEqual(ssim_single, ssim_batched[0])

  def test_ssim_video_2d(self):
    # make video from copying frames
    x_vid = np.repeat(self.vol1[:, 0:1], 5, axis=1)
    y_vid = np.repeat(self.vol2[:, 0:1], 5, axis=1)
    ssim_vid = metrics.ssim(x_vid, y_vid, dim=2, video=True)
    # make sure only batch dimension is preserved
    self.assertSequenceEqual(ssim_vid.shape, (self.vol1.shape[0],))
    x = self.vol1[:, 0]
    y = self.vol2[:, 0]
    ssim_img = metrics.ssim(x, y, dim=2, video=False)
    np.testing.assert_allclose(ssim_vid, ssim_img, atol=1e-6)

  def test_ssim_video_3d(self):
    # make volumetric video by repetition
    x_vid = np.repeat(np.expand_dims(self.vol1, axis=1), 5, axis=1)
    y_vid = np.repeat(np.expand_dims(self.vol2, axis=1), 5, axis=1)
    ssim_vid = metrics.ssim(x_vid, y_vid, dim=3, video=True)
    # make sure only batch dimension is preserved
    self.assertSequenceEqual(ssim_vid.shape, (self.vol1.shape[0],))
    ssim_img = metrics.ssim(self.vol1, self.vol2, dim=3, video=False)
    np.testing.assert_allclose(ssim_vid, ssim_img, atol=1e-6)

  def test_ssim_wo_channel(self):
    x = self.vol1[..., 0]
    y = self.vol2[..., 0]
    ssim_wo_channel = metrics.ssim(x, y, dim=3, has_channel=False)
    x, y = x[..., np.newaxis], y[..., np.newaxis]
    ssim_w_channel = metrics.ssim(x, y, dim=3, has_channel=True)
    np.testing.assert_allclose(ssim_wo_channel, ssim_w_channel)


class TraceMetricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.preds = np.array([
        [0.5, 0.3, 0.8, 0.2, 0.1],
        [0.4, 0.1, 0.2, 0.2, 0.0]
    ])
    self.targets = self.preds * 2
    self.segment_ids = np.array([0, 1, 2, 1, 0])
    self.elems_per_segment = np.array([2, 2, 1])

  def test_segment_mean_against_manual(self):
    data = self.preds[0]
    seg_mean = metrics._segment_mean(
        data, self.segment_ids, self.elems_per_segment
    )
    seg_mean_true = np.array([0.3, 0.25, 0.8])
    np.testing.assert_allclose(seg_mean_true, seg_mean)
    seg_mean_true_nonzero = seg_mean_true[1:]
    seg_mean_nonzero = metrics._segment_mean_non_background(
        data, self.segment_ids, self.elems_per_segment
    )
    np.testing.assert_allclose(seg_mean_true_nonzero, seg_mean_nonzero)

  @parameterized.named_parameters(
      ('with channel', True),
      ('without channel', False))
  def test_trace_based_metric_against_manual(self, has_channel):
    trace_metric = metrics.make_trace_based_metric(metrics.mae)
    if has_channel:
      preds = self.preds[..., np.newaxis]
      targets = self.targets[..., np.newaxis]
    else:
      preds, targets = self.preds, self.targets
    trace_mae = trace_metric(
        predictions=preds, targets=targets, trace_mask=self.segment_ids,
        trace_counts=self.elems_per_segment, has_channel=has_channel
    )
    trace_mae_true = np.array([0.525, 0.175])
    np.testing.assert_allclose(trace_mae_true, trace_mae)


if __name__ == '__main__':
  absltest.main()
