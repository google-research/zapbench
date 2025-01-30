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

"""Tests for data_loading."""

from absl.testing import absltest
from connectomics.jax.inputs import tensorloader as tl
import jax
import numpy as np
from zapbench import constants
from zapbench.video_forecasting import config
from zapbench.video_forecasting import data_loading as dl


class DataLoadingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    c = config.get_video_prediction_config()
    c.data_config.timesteps_input: int = 3
    c.data_config.timesteps_output: int = 2
    c.data_config.x_crop_size = None
    c.data_config.y_crop_size = None
    c.data_config.z_crop_size = None
    # simulate small setup with held out and pretraining splits
    constants.CONDITION_OFFSETS = (0, 52, 104, 156)
    constants.CONDITIONS_TRAIN = (0,)
    constants.CONDITIONS_HOLDOUT = (1,)
    constants.CONDITIONS = (0, 1)
    constants.MAX_CONTEXT_LENGTH = c.data_config.timesteps_input
    constants.PREDICTION_WINDOW_LENGTH = 2
    c.data_config.condition_offsets = constants.CONDITION_OFFSETS
    # by default use only train conditions
    c.data_config.conditions = constants.CONDITIONS_TRAIN
    c.num_epochs = 1
    c.global_batch_size = 3
    rng = np.random.default_rng(seed=0)

    ts_test_config = {
        'driver': 'array',
        'dtype': 'float32',
        'array': rng.uniform(0, 1, (8, 6, 4, 156)).tolist(),
    }
    c.data_config.tensorstore_input_config = ts_test_config
    c.data_config.tensorstore_output_config = ts_test_config
    ts_stim_config = {
        'driver': 'array',
        'dtype': 'float32',
        'array': rng.uniform(0, 1, (156, 3)).tolist(),
    }
    c.data_config.tensorstore_stimulus_config = ts_stim_config
    self.config = c
    self.frame_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    self.stim_sharding = self.frame_sharding
    self.lead_time_sharding = self.frame_sharding
    self.shardings = (
        self.frame_sharding,
        self.stim_sharding,
        self.lead_time_sharding,
    )
    self.args = (
        self.config.data_config,
        self.frame_sharding,
        self.stim_sharding,
        self.lead_time_sharding,
    )
    self.ts_out_config = {
        'driver': 'array',
        'dtype': 'float32',
        'array': rng.uniform(0, 1, (16, 12, 4, 156)).tolist(),
    }

  def test_data_source_split(self):
    data_source = dl.VideoTensorSource(*self.args, split='train')
    # 0.7 * 50 = 35 is volume shape and we have each 5 frames -> 31 data points
    self.assertLen(data_source, 31)
    data_source = dl.VideoTensorSource(*self.args, split='val')
    # 0.1 * 50 = 5: 1 data point + 3 due to shifted input context
    self.assertLen(data_source, 4)
    data_source = dl.VideoTensorSource(*self.args, split='test')
    # 0.2 * 50 = 10: 6 data points + 3 shifted
    self.assertLen(data_source, 9)
    data_source = dl.VideoTensorSource(*self.args, split='test_holdout')
    # no held out data in first trial
    self.assertEmpty(data_source)

  def test_data_source_split_holdout(self):
    self.config.data_config.conditions = (1,)
    for split in ['train', 'val', 'test']:
      data_source = dl.VideoTensorSource(*self.args, split=split)
      self.assertEmpty(data_source)
    data_source = dl.VideoTensorSource(*self.args, split='test_holdout')
    # contains all 50 timesteps with 5 consecutive steps -> 46 data points
    self.assertLen(data_source, 46)

  def test_data_source_split_pretrain(self):
    self.config.data_config.conditions = (2,)
    for split in ['val', 'test', 'test_holdout']:
      data_source = dl.VideoTensorSource(*self.args, split=split)
      self.assertEmpty(data_source)
    data_source = dl.VideoTensorSource(*self.args, split='train')
    # contains all 50 timesteps with 5 consecutive steps -> 46 data points
    self.assertLen(data_source, 46)

  def test_data_source_items(self):
    data_source = dl.VideoTensorSource(*self.args, split='val')
    self.assertEqual(
        data_source.item_shape['input_frames'],
        (self.config.data_config.timesteps_input, 8, 6, 4, 1),
    )
    self.assertEqual(
        data_source.item_shape['output_frames'],
        (self.config.data_config.timesteps_output, 8, 6, 4, 1),
    )
    self.assertEqual(
        data_source[0]['input_frames'].to_global().shape,
        (1, self.config.data_config.timesteps_input, 8, 6, 4, 1),
    )
    self.assertEqual(
        data_source[0]['output_frames'].to_global().shape,
        (1, self.config.data_config.timesteps_output, 8, 6, 4, 1),
    )
    self.assertEqual(
        data_source[0]['input_stimulus'].to_global().shape,
        (1, self.config.data_config.timesteps_input, 3),
    )
    self.assertEqual(
        data_source[0]['output_stimulus'].to_global().shape,
        (1, self.config.data_config.timesteps_output, 3),
    )
    ixs = tl.BatchMetadata(indices=[0, 1])
    self.assertEqual(
        data_source[ixs]['input_frames'].to_global().shape,
        (2, self.config.data_config.timesteps_input, 8, 6, 4, 1),
    )
    self.assertEqual(
        data_source[ixs]['output_frames'].to_global().shape,
        (2, self.config.data_config.timesteps_output, 8, 6, 4, 1),
    )
    self.assertEqual(
        data_source[ixs]['input_stimulus'].to_global().shape,
        (2, self.config.data_config.timesteps_input, 3),
    )
    self.assertEqual(
        data_source[ixs]['output_stimulus'].to_global().shape,
        (2, self.config.data_config.timesteps_output, 3),
    )

  def test_data_source_iterator(self):
    data_source = dl.VideoTensorSource(*self.args, split='val')
    items = []
    for elem in data_source:
      items.append(elem)
    self.assertLen(items, len(data_source))

  def test_data_source_crop(self):
    self.config.data_config.x_crop_size = 5
    self.config.data_config.y_crop_size = 4
    self.config.data_config.z_crop_size = 2
    data_source = dl.VideoTensorSource(*self.args, split='val')
    expected_shape = (self.config.data_config.timesteps_input, 5, 4, 2, 1)
    ix = tl.BatchMetadata(
        indices=[
            0,
        ]
    )
    self.assertEqual(
        data_source[ix]['input_frames'].to_global()[0].shape, expected_shape
    )
    self.assertEqual(data_source.item_shape['input_frames'], expected_shape)
    expected_shape = (self.config.data_config.timesteps_output, 5, 4, 2, 1)
    self.assertEqual(
        data_source[ix]['output_frames'].to_global()[0].shape, expected_shape
    )
    self.assertEqual(data_source.item_shape['output_frames'], expected_shape)
    np.testing.assert_array_equal(
        np.array(data_source[ix]['lead_time'].to_global()), np.zeros((1,))
    )

  def test_data_source_keep_time(self):
    self.config.data_config.timesteps_input = 1
    self.config.data_config.timesteps_output = 1
    data_source = dl.VideoTensorSource(*self.args, split='val')
    expected_shape = (1, 8, 6, 4, 1)
    ix = tl.BatchMetadata(indices=[0])
    self.assertEqual(
        data_source[ix]['input_frames'].to_global()[0].shape, expected_shape
    )
    self.assertEqual(data_source.item_shape['input_frames'], expected_shape)

  def test_loader(self):
    self.config.eval_pad_last_batch = True
    data_loader = dl.get_dataset(self.config, 1, *self.shardings, split='val')

    # test shape of full batch
    elem = next(iter(data_loader))
    batch_size = self.config.global_batch_size
    timesteps_input = self.config.data_config.timesteps_input
    timesteps_output = self.config.data_config.timesteps_output
    expected_shape_in = (batch_size, timesteps_input, 8, 6, 4, 1)
    self.assertEqual(elem['input_frames'].to_global().shape, expected_shape_in)
    expected_shape_stim_in = expected_shape_in[:2] + (3,)
    self.assertEqual(
        elem['input_stimulus'].to_global().shape, expected_shape_stim_in
    )
    expected_shape_out = (batch_size, timesteps_output, 8, 6, 4, 1)
    self.assertEqual(
        elem['output_frames'].to_global().shape, expected_shape_out
    )
    expected_shape_stim_out = expected_shape_out[:2] + (3,)
    self.assertEqual(
        elem['output_stimulus'].to_global().shape, expected_shape_stim_out
    )

    # test shape of last element in val set, should be one due to
    # data source length of 4 and batch_size 3.
    expected_shape_in = (1, *expected_shape_in[1:])
    for elem in data_loader:
      continue
    self.assertEqual(elem['input_frames'].to_global().shape, expected_shape_in)

  def test_timestep_boundaries(self):
    data_source = dl.VideoTensorSource(*self.args, split='train')
    self.assertEqual(len(data_source.timesteps), len(data_source.boundaries))
    # exclusive boundary should be timesteps_input + 1 away from start_timestep
    timesteps_model = (
        self.config.data_config.timesteps_input
        + self.config.data_config.timesteps_output
    )
    true_boundary = max(data_source.timesteps) + timesteps_model
    for boundary in data_source.boundaries:
      self.assertEqual(boundary, true_boundary)

  def test_time_conditioned_loader(self):
    self.config.data_config.timesteps_output = 1
    data_loader = dl.get_dataset(
        self.config, 1, *self.shardings, split='train', sample_lead_time=True
    )
    elem = next(iter(data_loader))
    self.assertLen(elem['lead_time'].to_global(), 3)
    # lead time should be at least 1 when used.
    self.assertNotIn(0, elem['lead_time'].to_global())

  def test_lead_time_sampling(self):
    self.config.data_config.timesteps_output = 1
    dl_cond = dl.get_dataset(
        self.config, 1, *self.shardings, split='train', sample_lead_time=True
    )
    elem_cond = next(iter(dl_cond))
    self.config.data_config.timesteps_output = 2
    dl_batch = dl.get_dataset(
        self.config, 1, *self.shardings, split='train', sample_lead_time=False
    )
    # samples should be equal so that lower lead times are not overrepresented.
    self.assertLen(dl_batch._tensor_source, len(dl_cond._tensor_source))
    elem_batch = next(iter(dl_batch))
    np.testing.assert_array_equal(
        elem_batch['input_frames'].to_global(),
        elem_cond['input_frames'].to_global(),
    )
    batch_elems = np.arange(self.config.global_batch_size)
    indices = elem_cond['lead_time'].to_global().flatten().astype(np.int32) - 1
    np.testing.assert_array_equal(
        elem_batch['output_frames'].to_global()[batch_elems, indices],
        elem_cond['output_frames'].to_global().squeeze(axis=1),
    )

  def test_determinism(self):
    self.config.data_config.timesteps_output = 1
    dl_a = dl.get_dataset(
        self.config, 1, *self.shardings, split='train', sample_lead_time=True
    )
    dl_b = dl.get_dataset(
        self.config, 1, *self.shardings, split='train', sample_lead_time=True
    )
    elem_a = next(iter(dl_a))
    elem_b = next(iter(dl_b))
    np.testing.assert_allclose(
        elem_a['lead_time'].to_global(), elem_b['lead_time'].to_global()
    )
    np.testing.assert_allclose(
        elem_a['output_frames'].to_global(), elem_b['output_frames'].to_global()
    )

  def test_separate_in_and_output_volumes(self):
    # 2 times upsampled in x, y, and same in z
    self.config.data_config.tensorstore_output_config = self.ts_out_config
    # crop from (16, 12, 4) to (12, 8, 4)
    self.config.data_config.x_crop_size = 12
    self.config.data_config.y_crop_size = 8
    self.config.data_config.z_crop_size = 2
    data_loader = dl.get_dataset(
        self.config, 1, *self.shardings, split='val', sample_lead_time=False
    )
    data_source = data_loader.tensor_source
    expected_in_shape = (self.config.data_config.timesteps_input, 6, 4, 2, 1)
    expected_out_shape = (self.config.data_config.timesteps_output, 12, 8, 2, 1)
    self.assertEqual(data_source.item_shape['input_frames'], expected_in_shape)
    self.assertEqual(
        data_source.item_shape['output_frames'], expected_out_shape
    )
    sample = next(iter(data_loader))
    self.assertEqual(
        sample['input_frames'].to_global().shape,
        (3, self.config.data_config.timesteps_input, 6, 4, 2, 1),
    )
    self.assertEqual(
        sample['output_frames'].to_global().shape,
        (3, self.config.data_config.timesteps_output, 12, 8, 2, 1),
    )


if __name__ == '__main__':
  absltest.main()
