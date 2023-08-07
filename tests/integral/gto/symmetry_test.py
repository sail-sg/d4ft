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

from absl.testing import absltest, parameterized

from d4ft.integral.gto import symmetry


class SymmetryTest(parameterized.TestCase):

  @parameterized.parameters(
    (6, True),
    (6, False),
  )
  def test_sym_idx(self, n_pgtos, four_c):
    if four_c:
      idx_counts = symmetry.get_4c_sym_idx(n_pgtos)
      print(idx_counts)
      self.assertEqual(len(idx_counts), symmetry.unique_ijkl(n_pgtos))
      self.assertEqual(idx_counts.shape[1], 5)

    else:
      idx_counts = symmetry.get_2c_sym_idx(n_pgtos)
      print(idx_counts)
      self.assertEqual(len(idx_counts), symmetry.unique_ij(n_pgtos))
      self.assertEqual(idx_counts.shape[1], 3)


if __name__ == "__main__":
  absltest.main()
