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

"""Config for video forecasting models."""

import dataclasses
from typing import Any

import ml_collections
from zapbench import constants
from zapbench.models import nlinear
from zapbench.models import nunet


def get_common_config() -> ml_collections.ConfigDict:
  """Returns a default config dict for experiments."""

  c = ml_collections.ConfigDict()
  c.optimizer = 'adamw'

  # For a single training example. The effective learning rate is
  # automatically (linearly) scaled as a function of the global
  # batch size.
  c.learning_rate = 0.001
  c.scale_learning_rate_by_global_batch_size = True
  c.min_learning_rate = 0.001
  c.num_warmup_steps = 5000
  c.scheduler = 'constant'

  c.num_epochs = int(1e6)
  c.per_device_batch_size = 8

  c.grad_clip_method = ''
  c.grad_clip_value = 0.0
  c.sgd_momentum = 0.9
  c.weight_decay = 0.0001
  c.num_train_steps = -1
  c.num_eval_steps = -1

  # If batches should be padded to evaluate the entire dataset.
  c.eval_pad_last_batch = True

  # How frequently to verify that the parameters are identical
  # across all workers.
  c.log_loss_every_steps = 250

  # ~15 min at 7.5 ex/s, which is the average training speed of a d=20
  # convstack model on 16xA100.
  c.checkpoint_every_steps = 6750

  # If specified, starts the experiment with the specified checkpoint
  # (from another experiment) instead of random weights.
  c.init_from_cpoint = ''

  # PRNG seed.
  c.seed = 42

  return c


@dataclasses.dataclass
class VideoPredictionDataConfig:
  """Config for video prediction data.

  Attributes:
    tensorstore_input_config: json string for opening input tensorstore
    tensorstore_output_config: json string for opening target tensorstore
    tensorstore_stimulus_config: json string for opening stimuli tensorstore
    eval_size: ratio of evaluation data between 0 and 1
    x_crop_size: size of center crop in x dimension
    y_crop_size: size of center crop in y dimension
    z_crop_size: size of center crop in z dimension
    timesteps_input: number of contiguous timesteps for input
    timesteps_output: number of contiguous timesteps to predict
    dynamic_range: min-max range of the data for metric normalization
    num_threads: number of threads for parallel prefetching
    condition_offsets: timestep indices where a condition starts and/or ends
    conditions: conditions to use for training/evaluation
  """

  tensorstore_input_config: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )
  tensorstore_output_config: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )
  tensorstore_stimulus_config: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )
  x_crop_size: int | None = 2048
  y_crop_size: int | None = 1152
  z_crop_size: int | None = 72
  timesteps_input: int = 8
  timesteps_output: int = 1
  dynamic_range: float = 1.0
  num_threads: int = 16
  condition_offsets: tuple[int, ...] = ()
  conditions: tuple[int, ...] = ()


def get_video_prediction_config() -> ml_collections.ConfigDict:
  """Default config for video prediction models."""
  c = get_common_config()
  c.data_config = ml_collections.ConfigDict(
      dataclasses.asdict(VideoPredictionDataConfig())
  )

  c.model_class = 'nunet.Nunet'
  c.nunet_config = ml_collections.ConfigDict(
      dataclasses.asdict(nunet.NunetConfig())
  )

  # to use nlinear, set config.model_class to 'nlinear.VideoNlinearUnivariate'
  # or 'nlinear.VideoNlinearGlobalUnivariate'
  c.nlinear_config = ml_collections.ConfigDict(
      dataclasses.asdict(nlinear.NlinearConfig())
  )
  c.nlinear_config.num_outputs = 32
  c.nlinear_config.normalization = True
  c.nlinear_config.constant_init = True

  c.data_config.condition_offsets = constants.CONDITION_OFFSETS
  c.data_config.conditions = constants.CONDITIONS

  # meshing and sharding settings
  # should be ordered by ease to distribute (batch -> spatial -> ..)
  c.mesh_shape = (1, 1, 1, 1)  # for the manuscript, (1, 1, 4, 4) was used
  c.mesh_names = ('batch', 't', 'x', 'y')
  # how batch dimensions map to mesh axes
  c.mesh_names_batch = ('batch', 't', 'x', 'y')
  c.mesh_names_stimulus = ('batch', 't')

  c.optimizer = 'adamw'
  c.weight_decay = 1e-5
  c.learning_rate = 1e-4
  c.min_learning_rate = 1e-7
  c.scheduler = 'cosine'
  c.global_batch_size = 1
  c.scale_learning_rate_by_global_batch_size = False
  c.num_train_steps = 250_000
  c.criterion = 'mae'  # 'mse', 'twohot', 'hlgauss'
  c.loss_mask = 'trace'
  c.input_mask = 'none'
  c.eval_with_train_step = False
  c.train_split = 'train'  # or train_val
  c.val_split = 'val'  # or test

  # Data config
  c.data_config.timesteps_input = 4
  c.data_config.timesteps_output = 1

  # config for nunet C=4
  # 4 maxvit_blocks, 'concat' combine type, 4 res blocks
  # embed_dim = 64,
  # conditioning true is best also
  # Nunet config
  c.nunet_config.activation = 'swish'
  c.nunet_config.remat = False  # set to True if oom
  c.nunet_config.enforce_sharding = True
  c.nunet_config.embed_dim = 128
  c.nunet_config.upsample_dim = 32
  c.nunet_config.num_res_blocks_in = 0
  c.nunet_config.num_res_blocks_out = 4
  c.nunet_config.norm = 'groupnorm_32'
  c.nunet_config.num_maxvit_blocks = 0
  c.nunet_config.maxvit_num_heads = 4  # embed_dim / maxvit_num_heads = 16
  c.nunet_config.output_type = 'features'
  c.nunet_config.num_outputs = 1  # features
  c.nunet_config.time_conditioning = True
  c.nunet_config.conditioning = False
  c.nunet_config.channel_dropout = 0.0
  c.nunet_config.spatial_dropout = 0.0
  c.nunet_config.attention_dropout = 0.0
  c.nunet_config.combine_type = 'add'  # can use 'concat' if fits in memory
  c.nunet_config.kernel_size = (3, 3, 3)
  c.nunet_config.maxvit_patch_size = (8, 9, 9)
  c.nunet_config.resample_factors = ()
  c.nunet_config.upsample_factors = ()
  c.nunet_config.num_res_blocks_up = 1

  # Logging and writing config
  c.log_loss_every_steps = 500
  c.eval_every_steps = 1000
  c.num_eval_steps = 500
  c.images_every_steps = 10000000
  c.checkpoint_every_steps = 1_000
  c.consistency_check_every_steps = 10000000
  c.use_statix = False

  # Segmentation for masked and trace-based metrics
  c.seg_tensorstore_config = ml_collections.ConfigDict({
      'driver': 'zarr3',
      'kvstore': {
          'driver': 'gcs',
          'bucket': 'zapbench-release',
          'path': 'volumes/20240930/segmentation'
      },
  })
  c.seg_shape = [2048, 1328, 72]
  c.seg_downsampling = [1, 1, 1]
  c.mask_tensorstore_config = ml_collections.ConfigDict({
      'driver': 'zarr3',
      'kvstore': {
          'driver': 'gcs',
          'bucket': 'zapbench-release',
          'path': (
              'volumes/20240930/mask/'
          ),
      },
  })
  c.mask_shape = [2048, 1328, 288]
  c.mask_downsampling = [1, 1, 4]

  # Additionally compute trace-based metrics
  c.detailed_metrics = True
  return c
