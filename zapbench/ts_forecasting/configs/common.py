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

"""Common configuration."""

from collections import abc
import random
from typing import Any, Optional, Sequence

from connectomics.jax import config_util
import ml_collections as mlc
from zapbench import constants
from zapbench import data_utils


def _get_specs(
    conditions: Sequence[int],
    num_timesteps_context: int,
    split: str,
    timeseries: str,
    covariate_series: str,
) -> Sequence[dict[str, Any]]:
  """Get specs."""
  specs = []
  for condition in conditions:
    specs.append({
        'timeseries': data_utils.adjust_spec_for_condition_and_split(
            spec=data_utils.get_spec(timeseries),
            condition=condition,
            split=split,
            num_timesteps_context=num_timesteps_context,
        ).to_json(),
        'covariates': data_utils.adjust_spec_for_condition_and_split(
            spec=data_utils.get_covariate_spec(covariate_series),
            condition=condition,
            split=split,
            num_timesteps_context=num_timesteps_context,
        ).to_json(),
    })
  return specs


def get_infer_sets(
    num_timesteps_context: int,
) -> Sequence[dict[str, int | str]]:
  """Get infer sets config."""
  sets = []
  for condition, split in [(t, 'test') for t in constants.CONDITIONS_TRAIN] + [
      (t, 'test_holdout') for t in constants.CONDITIONS_HOLDOUT
  ]:
    inclusive_min, exclusive_max = data_utils.adjust_condition_bounds_for_split(
        split,
        *data_utils.get_condition_bounds(condition),
        num_timesteps_context=num_timesteps_context,
    )
    sets.append({
        'name': f'{split}_condition_{condition}',
        'start_idx': inclusive_min,
        'num_windows': data_utils.get_num_windows(
            inclusive_min, exclusive_max, num_timesteps_context
        ),
    })
  return sets


def get_config(
    timesteps_input: int = constants.MAX_CONTEXT_LENGTH,
    timesteps_output: int = constants.PREDICTION_WINDOW_LENGTH,
    output_head: Optional[str] = None,
    num_classes: int = 1,
    train_conditions: str = config_util.sequence_to_string(
        constants.CONDITIONS_TRAIN, separator='+'
    ),
    runlocal: bool = False,
    timeseries: str = constants.TIMESERIES_NAME,
    covariate_series: str = constants.COVARIATE_SERIES_NAME,
    val_ckpt_every_steps: int = 250,
    log_loss_every_steps: int = 100,
    seed: int | tuple[int, int] | None = -1,
    **unused_kwargs,
) -> mlc.ConfigDict:
  """Default config.

  Args:
    timesteps_input: Number of input timesteps.
    timesteps_output: Number of output timesteps.
    output_head: Explicitly choose output head; automatically selected if None.
    num_classes: Number of classes for categorical output head.
    train_conditions: Conditions used for training, as string separated by '+'.
    runlocal: Whether running locally or remotely.
    timeseries: Name of the timeseries in `constants.SPECS`.
    covariate_series: Name of the covariates in `constants.COVARIATE_SPECS`.
    val_ckpt_every_steps: Frequency of validation/checkpointing.
    log_loss_every_steps: Frequency of logging loss.
    seed: Optional random seed. Uses a randomly generated seed if None or
      integer small than zero.

  Returns:
    A `mlc.ConfigDict` instance with the common configuration options.
  """
  c = mlc.ConfigDict()

  # Store/parse args
  c.output_head = output_head
  c.num_classes = num_classes
  c.timesteps_input = c.timesteps_input_infer = timesteps_input
  c.timesteps_output = c.timesteps_output_infer = timesteps_output
  c.train_conditions = config_util.string_to_sequence(
      train_conditions, separator='+'
  )
  c.runlocal = runlocal
  c.timeseries = timeseries
  c.covariate_series = covariate_series
  c.val_ckpt_every_steps = val_ckpt_every_steps

  # Validate
  assert c.timesteps_input <= constants.MAX_CONTEXT_LENGTH
  assert c.timesteps_output <= constants.PREDICTION_WINDOW_LENGTH
  assert constants.PREDICTION_WINDOW_LENGTH % c.timesteps_output == 0
  assert c.timeseries in constants.SPECS

  # Jax config
  c.optimizer = 'adamw'
  c.learning_rate = 0.001
  c.min_learning_rate = 0.001
  c.scheduler = 'constant'
  c.scale_learning_rate_by_global_batch_size = False
  c.num_epochs = 1 if c.runlocal else int(1e6)
  c.per_device_batch_size = 8

  c.grad_clip_method = ''
  c.grad_clip_value = 0.0
  c.sgd_momentum = 0.9
  c.weight_decay = 0.0001

  c.num_train_steps = -1

  c.early_stopping = True
  c.early_stopping_metric = 'val_loss'
  c.early_stopping_patience = 20
  c.early_stopping_min_delta = 0.0001

  c.max_runtime_stopping = False
  c.max_runtime = 3600.0 * 24 * 2  # 2 days

  # NOTE: Stopping on interrupt tags is not supported in open-source version.
  c.interrupt_tag_interval = 30.0  # 0.5 min

  if seed is None or (isinstance(seed, int) and seed < 0):
    # Generate a random seed.
    c.seed = random.randint(1, 2_147_483_646)
  else:
    c.seed = seed

  c.periodic_profiling = True
  c.periodic_nvidia_smi = 0  # If >0 called every N steps

  c.log_loss_every_steps = log_loss_every_steps
  c.checkpoint_every_steps = val_ckpt_every_steps

  # Data sources
  c.prefetch = True
  c.sequential_data_source = True

  # Data loading
  c.grain_num_workers = 0  # Number of workers for Grain loaders.

  # Data pre-processing
  c.pre_process_str = ''

  # Data batch processing
  c.batch_process_str = ''

  # Training
  c.train_specs = _get_specs(
      conditions=c.train_conditions,
      num_timesteps_context=c.timesteps_input,
      split='train',
      timeseries=c.timeseries,
      covariate_series=c.covariate_series,
  )

  # Validation
  c.val_specs = _get_specs(
      conditions=c.train_conditions,
      num_timesteps_context=c.timesteps_input,
      split='val',
      timeseries=c.timeseries,
      covariate_series=c.covariate_series,
  )
  c.num_val_steps = -1  # = 0 to disable, = -1 to iterate over val batches
  c.val_pad_last_batch = False
  c.val_every_steps = val_ckpt_every_steps

  # Inference
  c.prediction_window_length = constants.PREDICTION_WINDOW_LENGTH
  c.num_warmup_infer_steps = 0
  c.infer_spec = {
      'timeseries': data_utils.get_spec(c.timeseries).to_json(),
      'covariates': data_utils.get_covariate_spec(c.covariate_series).to_json(),
  }
  c.infer_sets = get_infer_sets(
      num_timesteps_context=c.timesteps_input_infer + c.num_warmup_infer_steps)
  c.infer_batching_str = (  # 1x...
      'expand_dims(keys=("timeseries_input","timeseries_output","covariates_input","covariates_output"),axis=0)'  # pylint: disable=line-too-long
  )
  c.infer_save_array = True
  # {workdir}, {step} will be replaced, if present in prefix:
  c.infer_prefix = 'file://{workdir}/inference/step/{step}'
  c.infer_with_carry = False

  # Series
  c.series_spec_shape = data_utils.get_spec(c.timeseries).shape
  c.series_shape = (1, timesteps_input, c.series_spec_shape[1])

  # Covariates
  c.covariates, c.covariates_shapes = tuple(), tuple()

  # Output head
  if c.output_head is None:
    c.head = 'deterministic_mae' if c.num_classes == 1 else 'categorical'
  else:
    c.head = c.output_head
  if c.head == 'categorical':
    assert c.num_classes > 1
    c.head_num_classes = c.num_classes
    assert c.timeseries in constants.MIN_MAX_VALUES
    c.head_lower, c.head_upper = constants.MIN_MAX_VALUES[c.timeseries]

  return c


def metrics(
    arg: Optional[str] = None,
) -> abc.Sequence[str] | abc.Sequence[tuple[str, str]]:
  """Returns metrics to be shown in Flatboard."""
  del arg
  # Equivalent to return (('step', 'train_loss'), ...)
  return (
      'train_loss',
      'train_loss_std',
      'train_learning_rate',
      'steps_per_sec',
      'uptime',
      'train_mae',
      'val_mae',
  )


def sweep_from_hparams_dict(
    hparams_dict: dict[str, Any],
    add: abc.Callable[..., None],
    arg: Optional[str] = None,
) -> None:
  """Starts work unit(s) with varying config args from dict of hyperparams."""
  parsed_arg = config_util.parse_arg(arg, sweep='', lazy=True)
  if parsed_arg.sweep in hparams_dict:
    for i, a in enumerate(hparams_dict[parsed_arg.sweep]):
      add(arg=','.join([f'{k}={v}' for k, v in a.items()]), tags=f'wu_{i+1}')
  else:
    add(
        arg=arg,
        tags=('wu_0',),
    )
