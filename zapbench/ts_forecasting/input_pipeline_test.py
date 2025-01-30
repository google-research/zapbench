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

"""Tests for input pipeline."""

from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from zapbench.ts_forecasting import input_pipeline
from zapbench.ts_forecasting.configs import common


def build_placeholder_spec(shape: Sequence[int]) -> dict[str, Any]:
  """Builds a placeholder spec containing all ones."""
  return {
      'driver': 'array',
      'dtype': 'float32',
      'array': np.ones(shape).tolist(),
  }


def get_specs_for_test(
    num_timesteps: int, num_features: int, num_dynamic_covariates: int
):
  """Returns specs for testing."""
  return {
      'covariates': build_placeholder_spec(
          (num_timesteps, num_dynamic_covariates)
      ),
      'timeseries': build_placeholder_spec((num_timesteps, num_features)),
  }


class InputPipelineTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='short_context',
          timesteps_input=4,
      ),
      dict(
          testcase_name='long_context',
          timesteps_input=256,
      ),
      dict(
          testcase_name='repeated_specs',
          num_spec_repetitions=2,
      ),
  )
  def test_create_datasets(
      self,
      timesteps_input: int = 4,
      timesteps_output: int = 32,
      num_timesteps: int = 1000,
      num_features: int = 16,
      num_dynamic_covariates: int = 2,
      num_spec_repetitions: int = 1,
  ):
    specs = get_specs_for_test(
        num_timesteps=num_timesteps,
        num_features=num_features,
        num_dynamic_covariates=num_dynamic_covariates,
    )

    config = common.get_config(
        timesteps_input=timesteps_input, timesteps_output=timesteps_output
    )
    config.train_specs = [specs] * num_spec_repetitions
    config.val_specs = [specs] * num_spec_repetitions

    config.per_device_batch_size = 8

    train_loader, num_train_records, val_loader, num_val_records = (
        input_pipeline.create_datasets(config, seed=1)
    )

    assert (
        num_train_records
        == (
            num_timesteps - config.timesteps_input - config.timesteps_output + 1
        )
        * num_spec_repetitions
    )
    assert num_val_records == num_train_records

    train_iter = iter(train_loader)
    batch = next(train_iter)
    assert (
        batch['timestep'].sum() > np.arange(config.per_device_batch_size).sum()
    )  # Shuffling
    assert batch['timeseries_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_features,
    )
    assert batch['timeseries_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_features,
    )
    assert batch['covariates_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_dynamic_covariates,
    )
    assert batch['covariates_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_dynamic_covariates,
    )

    val_iter = iter(val_loader)
    batch = next(val_iter)
    assert (
        batch['timestep'].sum() == np.arange(config.per_device_batch_size).sum()
    )  # No shuffling
    assert batch['timeseries_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_features,
    )
    assert batch['timeseries_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_features,
    )
    assert batch['covariates_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_dynamic_covariates,
    )
    assert batch['covariates_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_dynamic_covariates,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='short_context',
          timesteps_input=4,
      ),
      dict(
          testcase_name='long_context',
          timesteps_input=256,
      ),
  )
  def test_inference_source(
      self,
      timesteps_input: int = 4,
      timesteps_output: int = 32,
      num_timesteps: int = 1000,
      num_features: int = 16,
      num_dynamic_covariates: int = 2,
  ):
    specs = get_specs_for_test(
        num_timesteps=num_timesteps,
        num_features=num_features,
        num_dynamic_covariates=num_dynamic_covariates,
    )

    config = common.get_config(
        timesteps_input=timesteps_input, timesteps_output=timesteps_output
    )
    config.infer_spec = specs
    config.infer_batching_str = (
        'expand_dims(keys=("timeseries_input","timeseries_output"),axis=0)'
    )

    inference_source = input_pipeline.create_inference_source_with_transforms(
        config
    )

    assert len(inference_source) == (
        num_timesteps - config.timesteps_input - config.timesteps_output + 1
    )

    # Batch dimension of 1 due to infer_batching_str
    assert inference_source[0]['timeseries_input'].shape == (
        1,
        config.timesteps_input,
        num_features,
    )
    assert inference_source[0]['timeseries_output'].shape == (
        1,
        config.timesteps_output,
        num_features,
    )

    # No batch dimension
    assert inference_source[0]['covariates_input'].shape == (
        config.timesteps_input,
        num_dynamic_covariates,
    )
    assert inference_source[0]['covariates_output'].shape == (
        config.timesteps_output,
        num_dynamic_covariates,
    )


if __name__ == '__main__':
  absltest.main()
