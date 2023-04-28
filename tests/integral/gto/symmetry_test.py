from absl.testing import absltest, parameterized
from d4ft.integral.gto import symmetry


class SymmetryTest(parameterized.TestCase):

  @parameterized.parameters(
    (6, True),
    (6, False),
  )
  def test_sym_idx(self, n_gtos, four_c):
    if four_c:
      idx_counts = symmetry.get_4c_sym_idx(n_gtos)
      print(idx_counts)
      self.assertEqual(len(idx_counts), symmetry.unique_ijkl(n_gtos))
      self.assertEqual(idx_counts.shape[1], 5)

    else:
      idx_counts = symmetry.get_2c_sym_idx(n_gtos)
      print(idx_counts)
      self.assertEqual(len(idx_counts), symmetry.unique_ij(n_gtos))
      self.assertEqual(idx_counts.shape[1], 3)


if __name__ == "__main__":
  absltest.main()
