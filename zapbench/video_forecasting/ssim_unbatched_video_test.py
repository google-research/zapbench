# Copyright 2026 The Google Research Authors.
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

"""Unbatched-video SSIM must treat frames as independent spatial inputs."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from zapbench.video_forecasting import metrics


def videos(dim, channels, frames):
  spatial = (5, 6) if dim == 2 else (4, 5, 6)
  shape = (frames,) + spatial + ((2,) if channels else ())
  rng = np.random.default_rng(11)
  return (
      jnp.asarray(rng.random(shape), dtype=jnp.float32),
      jnp.asarray(rng.random(shape), dtype=jnp.float32),
  )


class UnbatchedVideoSsimTest(parameterized.TestCase):

  @parameterized.parameters(
      (2, False, 1),
      (2, True, 1),
      (2, False, 3),
      (2, True, 3),
      (3, False, 1),
      (3, True, 1),
      (3, False, 3),
      (3, True, 3),
  )
  def test_score_and_map_match_separate_frames(self, dim, channels, frames):
    prediction, target = videos(dim, channels, frames)
    options = dict(dim=dim, has_channel=channels, filter_size=3)
    expected_maps = jnp.stack([
        metrics.ssim(p, t, return_map=True, **options)
        for p, t in zip(prediction, target)
    ])
    for evaluate in (
        lambda p, t: metrics.ssim(p, t, video=True, return_map=True, **options),
        jax.jit(
            lambda p, t: metrics.ssim(
                p, t, video=True, return_map=True, **options
            )
        ),
    ):
      actual = evaluate(prediction, target)
      self.assertEqual(actual.shape, expected_maps.shape)
      np.testing.assert_allclose(actual, expected_maps, rtol=2e-5, atol=2e-6)
    actual_score = metrics.ssim(prediction, target, video=True, **options)
    self.assertEqual(actual_score.shape, ())
    np.testing.assert_allclose(
        actual_score, expected_maps.mean(), rtol=2e-5, atol=2e-6
    )
    batched = metrics.ssim(
        prediction[None], target[None], video=True, **options
    )
    np.testing.assert_allclose(actual_score, batched[0], rtol=2e-5, atol=2e-6)

  @parameterized.parameters(2, 3)
  def test_video_gradients_match_mean_of_frame_gradients(self, dim):
    prediction, target = videos(dim, True, 2)
    options = dict(dim=dim, filter_size=3)
    direct = lambda p: metrics.ssim(p, target, video=True, **options)
    reference = lambda p: jnp.mean(
        jnp.stack([metrics.ssim(a, b, **options) for a, b in zip(p, target)])
    )
    actual = jax.jit(jax.grad(direct))(prediction)
    expected = jax.grad(reference)(prediction)
    self.assertTrue(np.isfinite(actual).all())
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)

  def test_changing_one_frame_does_not_change_other_frame_maps(self):
    prediction, target = videos(2, True, 3)
    before = metrics.ssim(
        prediction, target, video=True, filter_size=3, return_map=True
    )
    changed = prediction.at[1].set(target[1])
    after = metrics.ssim(
        changed, target, video=True, filter_size=3, return_map=True
    )
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[2], before[2])
    np.testing.assert_allclose(after[1], 1.0, rtol=1e-5, atol=1e-6)

  def test_single_pixel_filter_preserves_frame_count(self):
    prediction, target = videos(2, False, 3)
    actual = metrics.ssim(
        prediction,
        target,
        video=True,
        has_channel=False,
        filter_size=1,
        return_map=True,
    )
    self.assertEqual(actual.shape, prediction.shape + (1,))


if __name__ == '__main__':
  absltest.main()
