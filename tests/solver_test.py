# Copyright 2023 Garena Online Private Limited
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

from typing import Tuple

from absl.testing import absltest, parameterized

from d4ft.config import get_config
from d4ft.solver.drivers import incore_cgto_direct_opt


class SolverTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", (-1., -1.1)),
    # ("o", (-72., -75.)),
    # ("h2o", (-74., -80.)),
  )
  def test_incore_sgd(
    self, system: str, energy_bounds: Tuple[float, float]
  ) -> None:
    cfg = get_config()
    cfg.mol_cfg.mol = system
    cfg.mol_cfg.basis = '6-31g'
    e_total = incore_cgto_direct_opt(cfg, basis_optim=False)
    upper_bound, lower_bound = energy_bounds
    self.assertTrue(e_total < upper_bound and e_total > lower_bound)


if __name__ == "__main__":
  absltest.main()
