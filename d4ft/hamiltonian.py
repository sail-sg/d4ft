import jax
import jax.numpy as jnp
import numpy as np
import pyscf

from d4ft.config import DFTConfig
from d4ft.mo_coeff_fn import get_mo_coeff_fn
from d4ft.types import Hamiltonian
from d4ft.xc import get_xc_intor


def calc_hamiltonian(mol: pyscf.gto.mole.Mole, cfg: DFTConfig) -> Hamiltonian:
  """Discretize the electron Hamiltonian in GTO basis (except for XC)."""
  ovlp = mol.intor('int1e_ovlp_sph')
  # TODO: refactor this part
  kin = mol.intor_symmetric('int1e_kin')
  ext = mol.intor_symmetric('int1e_nuc')
  eri = 0.5 * mol.intor('int2e')

  n_mos = len(ovlp)
  mo_ab_idx, mo_counts_ab = get_2c_idx(n_mos)
  mo_abcd_idx, mo_counts_abcd = get_4c_idx(n_mos)

  # reduce symmetry
  kin = kin[mo_ab_idx[:, 0], mo_ab_idx[:, 1]]
  ext = ext[mo_ab_idx[:, 0], mo_ab_idx[:, 1]]
  eri = eri[mo_abcd_idx[:, 0], mo_abcd_idx[:, 1], mo_abcd_idx[:, 2],
            mo_abcd_idx[:, 3]]

  mo_coeff_fn = get_mo_coeff_fn(mol, rks)

  xc_intor = get_xc_intor(mol, cfg.xc_type, cfg.quad_level)

  nuclei = {
    'loc': jnp.array(mol.atom_coords()),
    'charge': jnp.array(mol.atom_charges())
  }

  e_nuc = e_nuclear(nuclei)

  hamiltonian = Hamiltonian(
    ovlp,
    kin,
    ext,
    eri,
    mo_ab_idx,
    mo_counts_ab,
    mo_abcd_idx,
    mo_counts_abcd,
    xc_intor,
    nocc,
    mo_coeff_fn,
    nuclei,
    e_nuc,
  )

  return hamiltonian


def get_energy_fns(hamiltonian: Hamiltonian):

  def kin_fn(mo_coeff) -> float:
    rdm1 = mo_coeff.T @ mo_coeff
    rdm1_2c_ab = rdm1[hamiltonian.mo_ab_idx[:, 0], hamiltonian.mo_ab_idx[:, 1]]
    e_kin = jnp.sum(hamiltonian.kin * hamiltonian.mo_counts_ab * rdm1_2c_ab)
    return e_kin

  def ext_fn(mo_coeff) -> float:
    rdm1 = mo_coeff.T @ mo_coeff
    rdm1_2c_ab = rdm1[hamiltonian.mo_ab_idx[:, 0], hamiltonian.mo_ab_idx[:, 1]]
    e_ext = jnp.sum(hamiltonian.ext * hamiltonian.mo_counts_ab * rdm1_2c_ab)
    return e_ext

  def eri_fn(mo_coeff) -> float:
    rdm1 = mo_coeff.T @ mo_coeff
    rdm1_4c_ab = rdm1[hamiltonian.mo_abcd_idx[:, 0],
                      hamiltonian.mo_abcd_idx[:, 1],]
    rdm1_4c_cd = rdm1[hamiltonian.mo_abcd_idx[:, 2],
                      hamiltonian.mo_abcd_idx[:, 3],]
    e_eri = jnp.sum(
      hamiltonian.eri * hamiltonian.mo_counts_abcd * rdm1_4c_ab * rdm1_4c_cd
    )
    return e_eri

  def xc_fn(mo_coeff) -> float:
    e_xc = hamiltonian.xc_intor(mo_coeff)
    return e_xc

  return kin_fn, ext_fn, eri_fn, xc_fn
