from functools import partial

import jax
from absl.testing import absltest, parameterized
from d4ft.config import DFTConfig, OptimizerConfig
from d4ft.hamiltonian.cgto_intors import get_cgto_intor
from d4ft.hamiltonian.dft_cgto import dft_cgto
from d4ft.hamiltonian.ortho import qr_factor
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import incore_int_sym
from d4ft.integral.quadrature.grids import grids_from_pyscf_mol
from d4ft.sgd_solver import sgd_solver
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.xc import get_xc_intor
from jax_xc import lda_x


class SolverTest(parameterized.TestCase):

  @parameterized.parameters(
    ("h2", (-1., -1.1)),
    ("o", (-72., -75.)),
    ("h2o", (-74., -80.)),
  )
  def test_incore_sgd_dft(self, system, energy_bounds):
    basis = '6-31g'
    key = jax.random.PRNGKey(137)

    pyscf_mol = get_pyscf_mol(system, basis)
    mol = Mol.from_pyscf_mol(pyscf_mol)
    cgto = CGTO.from_mol(mol, use_hk=False)
    incore_energy_tensors = incore_int_sym(cgto)
    cgto_intor = get_cgto_intor(
      cgto, intor="obsa", incore_energy_tensors=incore_energy_tensors
    )
    grids_and_weights = grids_from_pyscf_mol(pyscf_mol, 1)
    xc_fn = get_xc_intor(grids_and_weights, cgto, lda_x)
    mo_coeff_fn = partial(mol.get_mo_coeff, rks=True, ortho_fn=qr_factor)
    H_fac = partial(dft_cgto, cgto, cgto_intor, xc_fn, mo_coeff_fn)
    e_total, _, _ = sgd_solver(DFTConfig(), OptimizerConfig(), H_fac, key)
    upper_bound, lower_bound = energy_bounds
    self.assertTrue(e_total < upper_bound and e_total > lower_bound)


if __name__ == "__main__":
  absltest.main()
