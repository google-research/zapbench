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

"""Utilities for instantiating models."""

from typing import Any, Type

from connectomics.jax.models import util
# pylint:disable=unused-import, g-importing-member
from connectomics.jax.models.util import get_config_name
from connectomics.jax.models.util import load_config
from connectomics.jax.models.util import save_config
# pylint:enable=unused-import, g-importing-member

import flax.linen as nn
import ml_collections

# pylint:disable=unused-import
from zapbench.models import naive
from zapbench.models import nlinear
from zapbench.models import nunet
from zapbench.models import tide
from zapbench.models import tsmixer
# pylint:enable=unused-import

DEFAULT_PKG = 'zapbench.models'


def class_from_name(
    model_class: str, default_packages: str = DEFAULT_PKG
) -> tuple[Type, Type]:  # pylint:disable=g-bare-generic
  return util.class_from_name(model_class, default_packages)


def model_from_config(
    config: ml_collections.ConfigDict, default_packages: str = DEFAULT_PKG
) -> nn.Module:
  return util.model_from_config(config, default_packages)


def model_from_name(
    model_class: str,
    model_name: str | None = None,
    default_packages: str = DEFAULT_PKG,
    **kwargs,
) -> nn.Module:
  return util.model_from_name(
      model_class, model_name, default_packages, **kwargs
  )


def model_from_dict_config(
    config: dict[str, Any], default_packages: str = DEFAULT_PKG
) -> nn.Module:
  return util.model_from_dict_config(config, default_packages)
