import jax.numpy as jnp
import pyscf

from d4ft.config import DFTConfig
from d4ft.integral.gto import symmetry
from d4ft.mo_coeff_fn import get_mo_coeff_fn
from d4ft.nuclear import e_nuclear
from d4ft.types import Hamiltonian
from d4ft.xc import get_xc_intor


def calc_hamiltonian(mol: pyscf.gto.mole.Mole, cfg: DFTConfig) -> Hamiltonian:
  """Discretize the electron Hamiltonian in GTO basis (except for XC)."""
  ovlp = mol.intor('int1e_ovlp_sph')

  # TODO: refactor this part
  kin = mol.intor_symmetric('int1e_kin')
  ext = mol.intor_symmetric('int1e_nuc')
  eri = 0.5 * mol.intor('int2e')

  nmo = len(ovlp)
  mo_ab_idx_counts = symmetry.get_2c_sym_idx(nmo)
  mo_abcd_idx_counts = symmetry.get_4c_sym_idx(nmo)

  # reduce symmetry
  kin = kin[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
  ext = ext[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
  eri = eri[mo_abcd_idx_counts[:, 0], mo_abcd_idx_counts[:, 1],
            mo_abcd_idx_counts[:, 2], mo_abcd_idx_counts[:, 3]]

  hamiltonian = Hamiltonian(
    ovlp,
    kin,
    ext,
    eri,
    mo_ab_idx_counts,
    mo_abcd_idx_counts,
    get_xc_intor(mol, cfg.xc_type, cfg.quad_level),
    get_mo_coeff_fn(mol, cfg.rks),
    e_nuclear(jnp.array(mol.atom_coords()), jnp.array(mol.atom_charges())),
  )

  return hamiltonian


def get_energy_fns(hamiltonian: Hamiltonian):

  def kin_fn(mo_coeff) -> float:
    rdm1 = mo_coeff.T @ mo_coeff
    rdm1_2c_ab = rdm1[hamiltonian.mo_ab_idx_counts[:, 0],
                      hamiltonian.mo_ab_idx_counts[:, 1]]
    counts = hamiltonian.mo_ab_idx_counts[:, 2]
    e_kin = jnp.sum(hamiltonian.kin * counts * rdm1_2c_ab)
    return e_kin

  def ext_fn(mo_coeff) -> float:
    rdm1 = mo_coeff.T @ mo_coeff
    rdm1_2c_ab = rdm1[hamiltonian.mo_ab_idx_counts[:, 0],
                      hamiltonian.mo_ab_idx_counts[:, 1]]
    counts = hamiltonian.mo_ab_idx_counts[:, 2]
    e_ext = jnp.sum(hamiltonian.ext * counts * rdm1_2c_ab)
    return e_ext

  def eri_fn(mo_coeff) -> float:
    rdm1 = mo_coeff.T @ mo_coeff
    rdm1_4c_ab = rdm1[hamiltonian.mo_abcd_idx_counts[:, 0],
                      hamiltonian.mo_abcd_idx_counts[:, 1]]
    rdm1_4c_cd = rdm1[hamiltonian.mo_abcd_idx_counts[:, 2],
                      hamiltonian.mo_abcd_idx_counts[:, 3]]
    counts = hamiltonian.mo_abcd_idx_counts[:, 4]
    e_eri = jnp.sum(hamiltonian.eri * counts * rdm1_4c_ab * rdm1_4c_cd)
    return e_eri

  def xc_fn(mo_coeff) -> float:
    e_xc = hamiltonian.xc_intor(mo_coeff)
    return e_xc

  return kin_fn, ext_fn, eri_fn, xc_fn
