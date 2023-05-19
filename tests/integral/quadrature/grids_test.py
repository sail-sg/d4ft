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

"""Test grids.py."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from absl.testing import absltest, parameterized
from pyscf import gto
from pyscf.dft.gen_grid import Grids

from d4ft.integral.quadrature.grids import DifferentiableGrids


class _GridsTest(parameterized.TestCase):

  @parameterized.parameters(
    "C 0 0 0; O 0 0 1.4",
    "Cl 0 0 0; Ag 0 0 4",
  )
  def test_alignment_with_pyscf(self, config) -> None:
    m = gto.M(atom=config, basis="sto-3g")

    g = Grids(m)
    g.level = 1
    g.alignment = 0
    g.build(sort_grids=False)
    c1, w1 = g.coords, g.weights

    dg = DifferentiableGrids(m)
    dg.level = 1
    c2, w2 = dg.build(m.atom_coords())

    np.testing.assert_allclose(c1, c2, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(w1, w2, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
  absltest.main()
