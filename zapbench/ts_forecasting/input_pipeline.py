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

"""Deterministic input pipeline."""

from typing import Any, Sequence

from connectomics.jax import grain_util
import grain.python as grain
import jax
import ml_collections as mlc
import numpy as np
import tensorstore as ts
from zapbench.ts_forecasting import data_source


FlatFeatures = grain_util.FlatFeatures


def _build_merged_data_source(
    series: dict[str, Any],
    timesteps_input: int,
    timesteps_output: int,
    prefetch: bool,
    transforms: Sequence[grain.MapTransform] = (),
    sequential: bool = True,
) -> data_source.MergedTensorStoreTimeSeries:
  """Builds MergedTensorStoreTimeSeries."""
  srcs = []
  for name, input_spec in series.items():
    srcs.append(
        data_source.TensorStoreTimeSeries(
            config=data_source.TensorStoreTimeSeriesConfig(
                input_spec=(
                    input_spec.to_dict()
                    if isinstance(input_spec, mlc.ConfigDict)
                    else input_spec
                ),
                timesteps_input=timesteps_input,
                timesteps_output=timesteps_output,
            ),
            prefetch=prefetch,
            prefix=name,
            transforms=transforms,
            sequential=sequential,
        )
    )
  return data_source.MergedTensorStoreTimeSeries(*srcs)


def get_num_train_steps(num_train_records: int, config: mlc.ConfigDict) -> int:
  """Calculates the total number of training steps."""
  if config.num_train_steps > 0:
    return config.num_train_steps
  # We first shard the data (shard by process_count), then combine all epochs,
  # batch for all local devices.
  # In all steps we would drop the remainder (hence the use of integer
  # devision).
  # When start_index is 0 the train_ds.cardinality() and num_train_steps should
  # be equivalent.
  return int(num_train_records // jax.process_count() * config.num_epochs) // (
      config.per_device_batch_size * jax.local_device_count()
  )


def get_all_ops() -> list[tuple[str, type[grain.Transformation]]]:
  return sum(
      map(
          grain_util.get_all_ops,
          [__name__, 'connectomics.jax.grain_util'],
      ),
      [],
  )


def create_datasets(
    config: mlc.ConfigDict,
    seed: int,
) -> tuple[grain.DataLoader, int, grain.DataLoader, int]:
  """Create Grain data loaders for training and validation.

  For the same seed and config this will return the same datasets.
  The user is responsible to save()/load() the dataset iterators (for training)
  or calling reset() to restart the iterator (for val).

  Args:
    config: Configuration to use.
    seed: Seed for shuffle and random operations in the training dataset.

  Returns:
    A tuple with the training dataset loader, the number of records in the
    training dataset, and validation dataset loaders, and the number of records
    in the validation dataset.
  """
  # Padding for last batch is not supported yet, remainders are dropped
  drop_remainder = True
  shard_options = grain.ShardByJaxProcess(drop_remainder=drop_remainder)
  if config.val_pad_last_batch:
    raise NotImplementedError(
        'BatchWithPadElements is not implemented in PyGrain yet.'
    )

  # Configurable pre-processing; before batching
  all_ops = get_all_ops()
  transformations = list(grain_util.parse(config.pre_process_str, all_ops))

  # The pipeline runs per process and loads data for local devices/accelerators.
  process_batch_size = jax.local_device_count() * config.per_device_batch_size
  batch_op = grain.Batch(
      batch_size=process_batch_size, drop_remainder=drop_remainder
  )
  transformations.append(batch_op)

  # Configurable batch-processing
  transformations += list(grain_util.parse(config.batch_process_str, all_ops))

  train_source = data_source.ConcatenatedTensorStoreTimeSeries(*[
      _build_merged_data_source(
          series=series,
          timesteps_input=config.timesteps_input,
          timesteps_output=config.timesteps_output,
          prefetch=config.prefetch,
          sequential=config.sequential_data_source,
      )
      for series in config.train_specs
  ])
  train_sampler = grain.IndexSampler(
      num_records=len(train_source),
      shuffle=True,
      seed=seed,
      num_epochs=config.num_epochs,
      shard_options=shard_options,
  )
  train_loader = grain.DataLoader(
      data_source=train_source,
      sampler=train_sampler,
      operations=transformations,
      worker_count=config.grain_num_workers,
  )

  val_source = data_source.ConcatenatedTensorStoreTimeSeries(*[
      _build_merged_data_source(
          series=series,
          timesteps_input=config.timesteps_input,
          timesteps_output=config.timesteps_output,
          prefetch=config.prefetch,
          sequential=config.sequential_data_source,
      )
      for series in config.val_specs
  ])
  val_sampler = grain.IndexSampler(
      num_records=len(val_source),
      shuffle=False,
      seed=None,
      num_epochs=1,
      shard_options=shard_options,
  )
  val_loader = grain.DataLoader(
      data_source=val_source,
      sampler=val_sampler,
      operations=transformations,
      # For now, we do not parallelize the validation, because there is a bug on
      # DataLoader.__iter__ when used with Jax.
      worker_count=0,
  )

  return train_loader, len(train_source), val_loader, len(val_source)


def create_inference_source_with_transforms(
    config: mlc.ConfigDict,
) -> data_source.MergedTensorStoreTimeSeries:
  """Creates inference data source with transforms."""
  # Configurable pre-processing
  all_ops = get_all_ops()
  transforms = list(grain_util.parse(config.pre_process_str, all_ops))

  # Batching
  transforms += list(grain_util.parse(config.infer_batching_str, all_ops))

  # Configurable batch-processing
  transforms += list(grain_util.parse(config.batch_process_str, all_ops))

  return _build_merged_data_source(
      series=config.infer_spec,
      timesteps_input=config.timesteps_input_infer,
      timesteps_output=config.timesteps_output_infer,
      prefetch=config.prefetch,
      transforms=transforms,
      sequential=config.sequential_data_source,
  )


def get_static_covariates(config: mlc.ConfigDict) -> None | np.ndarray:
  """Gets static covariates."""
  if 'covariates_static' not in config.covariates:
    return None
  if 'static_covariates_spec' in config:
    return (
        ts.open(ts.Spec(config.static_covariates_spec.to_dict()))
        .result()
        .read()
        .result()
    )
  else:
    raise NotImplementedError
