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

"""Utilities."""

import json
import math
import re
import subprocess
import time
from typing import Any, Optional, Sequence

from clu import metrics as clu_metrics
from connectomics.common import file
import connectomics.jax.metrics as metrics_lib
import distrax
from flax import struct
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pandas as pd
import tensorstore as ts

from zapbench import constants


class EarlyStoppingWithStep(struct.PyTreeNode):
  """Early stopping to avoid overfitting during training.

  This is a variant of `flax.training.early_stopping.EarlyStopping` that also
  tracks the step.

  The following example stops training early if the difference between losses
  recorded in the current epoch and previous epoch is less than 1e-3
  consecutively for 2 times:

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)
    for epoch in range(1, num_epochs+1):
      rng, input_rng = jax.random.split(rng)
      optimizer, train_metrics = train_epoch(
          optimizer, train_ds, config.batch_size, epoch, input_rng)
      _, early_stop = early_stop.update(train_metrics['loss'], epoch)
      if early_stop.should_stop:
        print('Met early stopping criteria, breaking...')
        break

  Attributes:
    min_delta: Minimum delta between updates to be considered an
        improvement.
    patience: Number of steps of no improvement before stopping.
    best_metric: Current best metric value.
    patience_count: Number of steps since last improving update.
    should_stop: Whether the training loop should stop to avoid
        overfitting.
  """
  min_delta: float = 0
  patience: int = 0
  best_metric: float = float('inf')
  best_step: int = 0
  patience_count: int = 0
  should_stop: bool = False

  def reset(self) -> 'EarlyStoppingWithStep':
    return self.replace(best_metric=float('inf'),
                        patience_count=0,
                        should_stop=False,
                        best_step=0)

  def update(
      self, metric: float, step: int) -> tuple[bool, 'EarlyStoppingWithStep']:
    """Update the state based on metric.

    Args:
      metric: Current metric value.
      step: Current step.

    Returns:
      A pair (has_improved, early_stop), where `has_improved` is True when there
      was an improvement greater than `min_delta` from the previous
      `best_metric` and `early_stop` is the updated `EarlyStop` object.
    """

    if (
        math.isinf(self.best_metric)
        or self.best_metric - metric > self.min_delta
    ):
      return True, self.replace(best_metric=metric,
                                patience_count=0,
                                best_step=step)
    else:
      should_stop = self.patience_count >= self.patience or self.should_stop
      return False, self.replace(patience_count=self.patience_count + 1,
                                 should_stop=should_stop)


class MaxRuntimeStopping(struct.PyTreeNode):
  """Max runtime stopping.

  Stops training early if a maximum runtime is exceeded.

  Attributes:
    max_runtime: Maximum runtime in seconds. Ignored is less/equal zero.
    should_stop: Whether the training loop should stop.
    reference_timestamp: Reference timestamps in seconds. Can be set using
      `set_reference_timestamp`.
  """

  max_runtime: float = 0.0
  reference_timestamp: float = 0.0
  should_stop: bool = False

  def reset(self) -> 'MaxRuntimeStopping':
    return self.replace(
        max_runtime=0.0,
        reference_timestamp=0.0,
        should_stop=False,
    )

  def set_reference_timestamp(
      self, timestamp: Optional[float] = None
  ) -> 'MaxRuntimeStopping':
    """Sets reference timestamp."""
    return (
        self.replace(reference_timestamp=timestamp)
        if timestamp is not None
        else self.replace(reference_timestamp=time.time())
    )

  def update(self) -> 'MaxRuntimeStopping':
    """Updates state."""
    if self.reference_timestamp <= 0.0:
      raise ValueError(
          'Reference timestamp needs to be positive. '
          + 'Consider setting it using `set_reference_timestamp`.'
      )
    elapsed_runtime = time.time() - self.reference_timestamp
    if self.max_runtime > 0.0 and elapsed_runtime > self.max_runtime:
      return self.replace(should_stop=True)
    else:
      return self


class TrackBestStep(struct.PyTreeNode):
  """Keeps track of step at which lowest loss was obtained.

  Attributes:
    best_metric: Current best metric value.
  """
  best_metric: float = float('inf')
  best_step: int = 0

  def reset(self) -> 'TrackBestStep':
    return self.replace(best_metric=float('inf'), best_step=0)

  def update(self, metric: float, step: int) -> 'TrackBestStep':
    """Update the state."""
    if metric < self.best_metric:
      return self.replace(best_metric=metric, best_step=step)
    else:
      return self


def get_bin_upper_bounds(
    lower: float, upper: float, num_bins: int, extend_upper_interval: bool
) -> jnp.ndarray:
  """Gets bin upper bounds."""
  bin_upper_bounds = jnp.histogram_bin_edges(
      a=jnp.array([]), bins=num_bins, range=(lower, upper)
  )[1:]
  if extend_upper_interval:
    bin_upper_bounds = bin_upper_bounds.at[-1].set(jnp.inf)
  return bin_upper_bounds


def get_digitize_bijector(
    lower: float, upper: float, num_classes: int
) -> distrax.Bijector:
  """Gets bijector for digitization."""
  forward_bins = get_bin_upper_bounds(
      lower=lower, upper=upper, num_bins=num_classes, extend_upper_interval=True
  )
  inverse_bins = (
      get_bin_upper_bounds(
          lower=lower,
          upper=upper,
          num_bins=num_classes,
          extend_upper_interval=False,
      )
      - (forward_bins[1] - forward_bins[0]) / 2.0
  )
  return distrax.Lambda(
      forward=lambda x: jnp.digitize(  # pylint: disable=g-long-lambda
          x=x, bins=forward_bins, right=False
      ).astype(jnp.float32),
      inverse=lambda y: inverse_bins[y.astype(jnp.int32)],
      forward_log_det_jacobian=jnp.zeros_like,
      inverse_log_det_jacobian=jnp.zeros_like,
      event_ndims_in=0,
      event_ndims_out=0,
      is_constant_jacobian=True,
  )


def get_one_hot_bijector(num_classes: int) -> distrax.Bijector:
  """Gets bijector for one-hot encoding."""
  return distrax.Lambda(
      forward=lambda x: jax.nn.one_hot(x[...], num_classes=num_classes),
      inverse=lambda y: jnp.argmax(y, axis=-1),
      forward_log_det_jacobian=jnp.zeros_like,
      inverse_log_det_jacobian=jnp.zeros_like,
      event_ndims_in=0,
      event_ndims_out=0,
      is_constant_jacobian=True,
  )


def get_metrics_collection(
    metric_names: Sequence[str], prefix: str = ''
) -> type[clu_metrics.Collection]:
  """Gets metrics collection by names with optional prefix."""
  metrics_dict = {}
  for name in metric_names:
    if name == 'loss':
      metrics_dict[name] = clu_metrics.Average.from_output('loss')
    elif name == 'loss_std':
      metrics_dict[name] = clu_metrics.Std.from_output('loss')
    elif name == 'learning_rate':
      metrics_dict[name] = clu_metrics.LastValue.from_output('learning_rate')
    elif name.endswith('_step'):
      metrics_dict[name] = metrics_lib.PerStepAverage.from_fun(
          metrics_lib.make_per_step_metric(
              getattr(metrics_lib, name.removesuffix('_step'))
          )
      )
    else:
      metrics_dict[name] = clu_metrics.Average.from_fun(
          getattr(metrics_lib, name)
      )
  return metrics_lib.get_metrics_collection_from_dict(metrics_dict, prefix)


def create_ts_spec(
    path: str,
    shape: Sequence[int],
    dtype: str,
    blocksize: Optional[Sequence[int]] = None,
    compression: Optional[dict[str, Any]] = None,
    delete_existing: bool = True,
    driver: str = 'zarr',
) -> ts.TensorStore:
  """Creates a TensorStore."""
  spec = {
      'create': True,
      'delete_existing': delete_existing,
      'driver': driver,
      'kvstore': path,
  }
  if driver == 'n5':
    spec['metadata'] = {
        'dimensions': shape,
        'blockSize': shape if blocksize is None else blocksize,
        'dataType': dtype,
        'compression': {'type': 'raw'} if compression is None else compression,
    }
  elif driver == 'zarr':
    spec['metadata'] = {
        'shape': shape,
        'chunks': shape if blocksize is None else blocksize,
        'dtype': dtype,
        'compressor': compression,
    }
  else:
    raise ValueError(f'Unsupported driver: {driver}')
  return spec


def save_array(
    arr: np.ndarray | jnp.ndarray,
    path: str,
    blocksize: Optional[Sequence[int]] = None,
    compression: Optional[dict[str, Any]] = None,
    delete_existing: bool = True,
    driver: str = 'zarr',
):
  """Saves a numpy or jax array to a TensorStore."""
  spec = create_ts_spec(
      path,
      arr.shape,
      arr.dtype.str if driver == 'zarr' else str(arr.dtype),
      blocksize,
      compression,
      delete_existing,
      driver,
  )
  ds = ts.open(spec).result()
  ds[...] = arr


def load_array(path: str, driver: str = 'zarr') -> np.ndarray:
  """Loads a numpy array from a TensorStore."""
  ds = ts.open({
      'open': True,
      'driver': driver,
      'kvstore': path,
  }).result()
  return ds[...].read().result()


def try_nvidia_smi() -> Optional[str]:
  try:
    return subprocess.check_output(['nvidia-smi']).decode()
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def get_condition_number_from_string(
    string: str, pattern: str = r'.*condition_(\d+)[^\d]*'
) -> int:
  """Extracts condition number from a string."""
  matches = re.findall(pattern, string)
  if len(matches) == 1:
    try:
      return int(matches[0])
    except ValueError:
      return -1
  else:
    return -1


def get_method_from_config(config: ml_collections.ConfigDict) -> str:
  """Extracts method name from config."""
  if config.model_class == 'naive.MeanBaseline':
    return 'mean'
  elif config.model_class == 'naive.ReturnCovariates':
    return 'stimulus'
  elif config.model_class == 'nlinear.Nlinear':
    return 'linear'
  elif config.model_class == 'tide.Tide':
    return 'tide'
  elif config.model_class == 'tsmixer.Tsmixer':
    if not config.tsmixer_config.time_mix_only:
      return 'tsmixer'
    else:
      return 'time-mix'
  else:
    return config.model_class


def get_per_step_metrics_from_dict(
    results_dict: dict[str, float], metric: str, include_key: bool = False
) -> pd.DataFrame:
  """Extracts per-step metrics from results dictionary."""
  rows = []
  for k in results_dict.keys():
    if 'step' not in k or '/' not in k or metric.lower() not in k.lower():
      continue
    _, steps_ahead = k.split('/')
    row = {
        'steps_ahead': int(steps_ahead),
        metric: results_dict[k],
    }
    if include_key:
      row['key'] = k
    rows.append(row)
  return pd.DataFrame(rows)


def get_per_step_metrics_from_directory(
    directory: str,
    metric: str = 'MAE',
    include_condition: bool = True,
    include_file: bool = True,
    include_key: bool = True,
) -> pd.DataFrame:
  """Gets per-step results from all json-files in a directory."""
  dfs = []
  for json_file in [
      f for f in file.Path(directory).iterdir() if str(f).endswith('.json')
  ]:
    with file.Path(f'{directory}/{json_file}').open('rt') as f:
      loaded_json_file = json.loads(f.read())
      df = get_per_step_metrics_from_dict(
          loaded_json_file, metric=metric, include_key=include_key
      )
      if include_file:
        df['file'] = f'{directory}/{json_file}'
      if include_condition:
        df['condition'] = constants.CONDITION_NAMES[
            get_condition_number_from_string(str(json_file))
        ]
      dfs.append(df)
  return pd.concat(dfs)
