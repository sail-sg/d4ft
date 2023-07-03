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

from pathlib import Path
import os

import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from d4ft.system.crystal import Crystal


class CrystalTest(parameterized.TestCase):

  @parameterized.parameters(("C2", 3.5667, 76.5484253352856),)
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

    self.assertAlmostEqual(crystal.vol, expected_vol)

  @parameterized.parameters(("diamond.xyz", 76.5484253352856),)
  def test_init_from_xyz_file(self, xyz_file: str, expected_vol: float):
    runfiles_dir = Path(os.environ['TEST_SRCDIR'])
    logging.info(runfiles_dir)
    xyz = runfiles_dir / '__main__/d4ft/system/xyz_files' / xyz_file
    crystal = Crystal.from_xyz_file(xyz)

    logging.info(f'cell volume:{crystal.vol}')

    self.assertAlmostEqual(crystal.vol, expected_vol)


if __name__ == "__main__":
  absltest.main()
