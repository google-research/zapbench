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

"""Data utilities."""

import json
from typing import Optional

from connectomics.common import file
import numpy as np
import pandas as pd
import tensorstore as ts
from zapbench import constants


def get_spec(spec_name: str) -> ts.Spec:
  """Gets TensorStore spec from SPECS.

  Args:
    spec_name: Key in SPECS.

  Returns:
    TensorStore Spec.
  """
  if spec_name not in constants.SPECS:
    raise ValueError(f'{spec_name} not in {constants.SPECS.keys()}.')
  return ts.Spec(constants.SPECS[spec_name])


def get_covariate_spec(spec_name: str) -> ts.Spec:
  """Gets TensorStore spec from COVARIATE_SPECS.

  Args:
    spec_name: Key in COVARIATE_SPECS.

  Returns:
    TensorStore Spec.
  """
  if spec_name not in constants.COVARIATE_SPECS:
    raise ValueError(f'{spec_name} not in {constants.COVARIATE_SPECS.keys()}.')
  return ts.Spec(constants.COVARIATE_SPECS[spec_name])


def get_position_embedding_spec(spec_name: str) -> ts.Spec:
  """Gets TensorStore spec from POSITION_EMBEDDING_SPECS.

  Args:
    spec_name: Key in POSITION_EMBEDDING_SPECS.

  Returns:
    TensorStore Spec.
  """
  if spec_name not in constants.POSITION_EMBEDDING_SPECS:
    raise ValueError(
        f'{spec_name} not in {constants.POSITION_EMBEDDING_SPECS.keys()}.')
  return ts.Spec(constants.POSITION_EMBEDDING_SPECS[spec_name])


def get_rastermap_spec(spec_name: str) -> ts.Spec:
  """Gets TensorStore spec from RASTERMAP_SPECS.

  Args:
    spec_name: Key in RASTERMAP_SPECS.

  Returns:
    TensorStore Spec.
  """
  if spec_name not in constants.RASTERMAP_SPECS:
    raise ValueError(f'{spec_name} not in {constants.RASTERMAP_SPECS.keys()}.')
  return ts.Spec(constants.RASTERMAP_SPECS[spec_name])


def get_segmentation_dataframe(df_name: str) -> pd.DataFrame:
  """Gets segmentation dataframe from SEGMENTATION_DATAFRAMES.

  Can be mapped to segmentations through the label-column and to indices in
  associated trace matrices though `df.loc[trace_idx]` (or label minus 1).

  Args:
    df_name: Key in SEGMENTATION_DATAFRAMES.

  Returns:
    Dataframe.
  """
  path = constants.SEGMENTATION_DATAFRAMES[df_name]
  with file.Path(path).open('r') as f:
    df = pd.DataFrame(json.load(f))
  return df.sort_values('label').reset_index(drop=True)


def get_condition_bounds(condition: int) -> tuple[int, int]:
  """Get bounds of a condition's temporal indices while accounting for padding.

  Args:
    condition: Condition number, starting from zero.

  Returns:
    (inclusive_min, exclusive_max)-tuple of boundaries.
  """
  if condition < 0 or condition >= len(constants.CONDITION_OFFSETS) - 1:
    raise ValueError(
        f'condition must be in [0, {len(constants.CONDITION_OFFSETS)-1}]'
    )
  return (
      constants.CONDITION_OFFSETS[condition] + constants.CONDITION_PADDING,
      constants.CONDITION_OFFSETS[condition + 1] - constants.CONDITION_PADDING,
  )


def adjust_condition_bounds_for_split(
    split: str,
    inclusive_min: int,
    exclusive_max: int,
    num_timesteps_context: int,
) -> tuple[int, int]:
  """Adjust condition bounds for a split.

  Args:
    split: Requested split, 'train' (training set), 'val' (validation set),
      'test' (test set), 'test_holdout' (test set for a condition that was
      held-out from training), 'train_val' (train and validation set), or
      None (in which case the condition is not split).
    inclusive_min: Lower bound.
    exclusive_max: Upper bound.
    num_timesteps_context: Number of timesteps for context.

  Returns:
    (inclusive_min, exclusive_max)-tuple of boundaries.
  """
  num_timesteps_total = exclusive_max - inclusive_min
  num_timesteps_test = int(num_timesteps_total * constants.TEST_FRACTION)
  num_timesteps_val = int(num_timesteps_total * constants.VAL_FRACTION)
  num_timesteps_train = (
      num_timesteps_total - num_timesteps_test - num_timesteps_val
  )

  if split == 'train':
    exclusive_max = inclusive_min + num_timesteps_train
  elif split == 'val':
    exclusive_max = inclusive_min + num_timesteps_train + num_timesteps_val
    inclusive_min += num_timesteps_train - num_timesteps_context
  elif split == 'train_val':
    exclusive_max = inclusive_min + num_timesteps_train + num_timesteps_val
  elif split == 'test':
    inclusive_min += (
        num_timesteps_train + num_timesteps_val - num_timesteps_context
    )
  elif split == 'test_holdout':
    inclusive_min += constants.MAX_CONTEXT_LENGTH - num_timesteps_context
  elif split is None:
    pass
  else:
    raise ValueError(
        f'split must be train, val, test, test_holdout or None but is {split}.'
    )

  assert (
      inclusive_min >= 0
  ), f'inclusive_min must be >= 0 but is {inclusive_min}.'
  assert inclusive_min < exclusive_max, 'inclusive_min must be < exclusive_max.'

  return inclusive_min, exclusive_max


def get_num_windows(
    inclusive_min: int, exclusive_max: int, num_timesteps_context: int
) -> int:
  """Given condition bounds, calculate number of prediction windows at stride 1."""
  return (
      exclusive_max
      - inclusive_min
      - constants.PREDICTION_WINDOW_LENGTH
      - num_timesteps_context
      + 1
  )


def adjust_spec_for_condition_and_split(
    spec: ts.Spec,
    condition: int,
    split: Optional[str],
    num_timesteps_context: int,
) -> ts.Spec:
  """Adjust spec for condition and split.

  For example, to get the training timeseries for the first condition for an
  algorithm that uses 32 timesteps as context:
    spec = get_spec('timeseries')
    ds = ts.open(adjust_spec_for_condition_and_split(
        spec, 0, 'train', 32)).result()

  Args:
    spec: TensorStore spec with dimension `t`.
    condition: Condition number, starting from zero.
    split: Requested split, 'train' (training set), 'val' (validation set),
      'test' (test set), 'test_holdout' (test set for a condition that was
      held-out from training), or None (in which case the condition is not
      split).
    num_timesteps_context: Number of additional timesteps for context.

  Returns:
    TensorStore Spec.
  """
  if 't' not in spec.domain.labels:
    raise ValueError('Required dimension label `t` not found in spec.')

  inclusive_min, exclusive_max = adjust_condition_bounds_for_split(
      split, *get_condition_bounds(condition), num_timesteps_context
  )

  return spec[ts.d['t'][slice(inclusive_min, exclusive_max)]].translate_to[0]


def get_rastermap_indices(timeseries: str) -> np.ndarray:
  """Gets rastermap indices."""
  return json.loads(
      file.Path(
          constants.RASTERMAP_SORTINGS[timeseries],
      ).read_text('rt')
  )


def get_indices_to_invert_rastermap_sorting(timeseries: str) -> np.ndarray:
  """Gets indices that invert rastermap sorting."""
  return np.argsort(get_rastermap_indices(timeseries))
