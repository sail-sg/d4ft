from functools import partial

import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from d4ft.hamiltonian.ortho import qr_factor
from d4ft.integral.pw.pw import PW
from d4ft.system.crystal import Crystal
from d4ft.utils import to_3dvec
import haiku as hk


class PWTest(parameterized.TestCase):

  @parameterized.parameters(
    ("C2", 3.5667, 76.5484253352856),
  )
  def test_init_from_name_and_lattice(
    self, crystal_name: str, a: float, expected_vol: float
  ):
    position = np.array([[0, 0, 0], [a / 4, a / 4, a / 4]])
    cell = np.array(
      [
        [0., a / 2, a / 2],
        [a / 2, 0, a / 2],
        [a / 2, a / 2, 0.],
      ]
    )

    crystal = Crystal.from_name_and_lattice(crystal_name, position, cell)
    logging.info(f'cell volume:{crystal.vol}')

    pw = PW.from_crystal(
      crystal=crystal,
      n_g_pts=to_3dvec(16, int),
      n_k_pts=to_3dvec(1, int),
      e_cut=10.0
    )

    pw_coeff_fn = hk.without_apply_rng(
      hk.transform(pw.get_pw_coeff(polarized=True, ortho_fn=qr_factor))
    )
    params = pw_coeff_fn.init()
    pw_coeff = pw_coeff_fn.apply(params)
    nr, nG = pw.eval(pw_coeff)
    logging.info(nr.shape)
    logging.info(nG.shape)


if __name__ == "__main__":
  absltest.main()
