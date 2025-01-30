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

"""Tests for the util module."""

from absl.testing import absltest
from absl.testing import parameterized

from zapbench.models import util


class UtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mean_baseline',
          class_name='naive.MeanBaseline',
          expected_config_name='mean_baseline_config',
      ),
      dict(
          testcase_name='return_covariates',
          class_name='naive.ReturnCovariates',
          expected_config_name='return_covariates_config',
      ),
      dict(
          testcase_name='nlinear',
          class_name='nlinear.Nlinear',
          expected_config_name='nlinear_config',
      ),
      dict(
          testcase_name='nunet',
          class_name='nunet.Nunet',
          expected_config_name='nunet_config',
      ),
      dict(
          testcase_name='tide',
          class_name='tide.Tide',
          expected_config_name='tide_config',
      ),
      dict(
          testcase_name='tsmixer',
          class_name='tsmixer.Tsmixer',
          expected_config_name='tsmixer_config',
      ),
  )
  def test_get_config_name(self, class_name: str, expected_config_name: str):
    _, cfg_cls = util.class_from_name(class_name)
    self.assertEqual(
        expected_config_name, util.get_config_name(cfg_cls.__name__))


if __name__ == '__main__':
  absltest.main()
