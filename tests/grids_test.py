"""Test grids.py."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf import gto
from pyscf.dft.gen_grid import Grids
from d4ft.grids import DifferentiableGrids
from absl.testing import absltest, parameterized


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
