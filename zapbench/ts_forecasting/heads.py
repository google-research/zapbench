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

"""Heads combine losses, metrics, and distributions."""

import dataclasses
from typing import Any

from clu import metrics as clu_metrics
import connectomics.jax.metrics as metrics_lib
import distrax
import jax.numpy as jnp
import ml_collections as mlc
import optax

from zapbench.ts_forecasting import util


@dataclasses.dataclass
class Head:
  """Generic head."""

  metrics: dict[str, type[clu_metrics.Collection]] = dataclasses.field(
      default_factory=dict
  )

  def compute_loss(
      self, predictions: jnp.ndarray, targets: jnp.ndarray
  ) -> jnp.ndarray:
    raise NotImplementedError

  def get_distribution(self, predictions: jnp.ndarray) -> distrax.Distribution:
    raise NotImplementedError


@dataclasses.dataclass
class DeterministicHead(Head):
  """Head with deterministic distribution."""

  loss_fn: Any = metrics_lib.mse

  def compute_loss(
      self, predictions: jnp.ndarray, targets: jnp.ndarray
  ) -> jnp.ndarray:
    return jnp.mean(self.loss_fn(predictions, targets))

  def get_distribution(self, predictions: jnp.ndarray) -> distrax.Distribution:
    return distrax.Deterministic(predictions)


def _reshape_predictions_for_logits(
    predictions: jnp.ndarray, num_classes: int
) -> jnp.ndarray:
  """Reshapes predictions for usage as logits; Bx(T'*C)xF -> BxT'xFxC."""
  return predictions.reshape(
      predictions.shape[0], -1, num_classes, predictions.shape[-1]
  ).transpose([0, 1, 3, 2])


@dataclasses.dataclass
class CategoricalHead(Head):
  """Head with one-hot categorical distribution."""

  lower: float = 0.0
  upper: float = 1.0
  num_classes: int = 64

  loss_fn: Any = optax.softmax_cross_entropy

  # Predictions are reshaped Bx(T'*C)xF -> BxT'xFxC by default.
  reshape_predictions: Any = _reshape_predictions_for_logits

  def __post_init__(self):
    # Targets are processed by the chain of bijectors below. Since bijectors are
    # applied in sequence starting from the end of the list, digitization is
    # followed by one-hot encoding, s.t. BxT'xF -> BxT'xFxC (forward direction).
    self._targets_chain = distrax.Chain([
        distrax.Block(util.get_one_hot_bijector(self.num_classes), ndims=3),
        distrax.Block(
            util.get_digitize_bijector(
                self.lower, self.upper, self.num_classes
            ),
            ndims=3,
        ),
    ])

  def compute_loss(
      self, predictions: jnp.ndarray, targets: jnp.ndarray
  ) -> jnp.ndarray:
    return jnp.mean(
        self.loss_fn(
            logits=self.reshape_predictions(predictions, self.num_classes),
            labels=self._targets_chain.forward(targets),
        )
    )

  def get_distribution(self, predictions: jnp.ndarray) -> distrax.Distribution:
    return distrax.Transformed(
        distrax.Independent(
            distrax.OneHotCategorical(
                logits=self.reshape_predictions(predictions, self.num_classes)
            ),
            reinterpreted_batch_ndims=2,
        ),
        distrax.Inverse(self._targets_chain),
    )


def create_head(config: mlc.ConfigDict) -> Head:
  """Creates head."""
  loss_metrics = ['loss', 'loss_std']

  if config.head.startswith('deterministic'):
    if config.head.endswith('mae'):
      loss_fn = metrics_lib.mae
    elif config.head.endswith('mse'):
      loss_fn = metrics_lib.mse
    else:
      loss_fn = metrics_lib.mae

    # TODO(jan-matthis): Consider tracking more metrics.
    common_metrics = ['mse', 'mse_step', 'mae', 'mae_step']
    metrics = {}
    metrics['train'] = util.get_metrics_collection(
        loss_metrics
        + [
            'learning_rate',
        ]
        + common_metrics,
        prefix='train_',
    )
    metrics['val'] = util.get_metrics_collection(
        loss_metrics + common_metrics, prefix='val_'
    )
    for infer_set in config.infer_sets:
      metrics[f'infer_{infer_set["name"]}'] = util.get_metrics_collection(
          common_metrics, prefix=f'infer_{infer_set["name"]}_'
      )

    return DeterministicHead(loss_fn=loss_fn, metrics=metrics)

  elif config.head == 'categorical':
    # NOTE: Sampling is only performed during inference; this restricts metrics
    # computation during train/val.
    # TODO(jan-matthis): Consider tracking probabilitic metrics.
    infer_metrics = ['mse', 'mse_step', 'mae', 'mae_step']
    metrics = {}
    metrics['train'] = util.get_metrics_collection(
        loss_metrics
        + [
            'learning_rate',
        ],
        prefix='train_',
    )
    metrics['val'] = util.get_metrics_collection(loss_metrics, prefix='val_')
    for infer_set in config.infer_sets:
      metrics[f'infer_{infer_set["name"]}'] = util.get_metrics_collection(
          infer_metrics, prefix=f'infer_{infer_set["name"]}_'
      )
    return CategoricalHead(
        metrics=metrics,
        lower=config.head_lower,
        upper=config.head_upper,
        num_classes=config.head_num_classes,
    )

  else:
    raise ValueError(f'Unknown head: {config.head}.')
