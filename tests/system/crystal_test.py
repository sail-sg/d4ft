from pathlib import Path
import os

import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from d4ft.system.crystal import Crystal


class CrystalTest(parameterized.TestCase):

  @parameterized.parameters(
    ("C2", 3.5667, 76.5484253352856),
  )
  def test_init_from_name_and_lattice(
    self, crystal_name: str, a: float, expected_vol: float
  ):
    position = np.array([[0, 0, 0], [a / 4, a / 4, a / 4]])
    cell = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    # n_g_pts = to_3dvec(4, dtype=int)
    # n_k_pts = to_3dvec(4, dtype=int)

    crystal = Crystal.from_name_and_lattice(crystal_name, position, cell)
    logging.info(f'cell volume:{crystal.vol}')

    self.assertAlmostEqual(crystal.vol, expected_vol)

  @parameterized.parameters(
    ("diamond.xyz", 76.5484253352856),
  )
  def test_init_from_xyz_file(self, xyz_file: str, expected_vol: float):
    runfiles_dir = Path(os.environ['TEST_SRCDIR'])
    logging.info(runfiles_dir)
    xyz = runfiles_dir / '__main__/d4ft/system/xyz_files' / xyz_file
    crystal = Crystal.from_xyz_file(xyz)

    logging.info(f'cell volume:{crystal.vol}')

    self.assertAlmostEqual(crystal.vol, expected_vol)


if __name__ == "__main__":
  absltest.main()
