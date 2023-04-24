import jax
import jax.numpy as jnp
import numpy as np
import pyscf

from d4ft.config import DFTConfig
from d4ft.orbitals import get_occupation_mask, qr_factor_param
from d4ft.types import Hamiltonian
from d4ft.xc import get_xc_intor


def get_2c_combs(n_orbs: int):
  """2-fold symmetry (a|b)"""
  ab_idx = np.vstack(np.triu_indices(n_orbs)).T
  offdiag_ab = ab_idx[:, 0] != ab_idx[:, 1]
  counts_ab = offdiag_ab + np.ones(len(ab_idx))
  return ab_idx, counts_ab


def get_4c_combs(n_orbs: int):
  """8-fold symmetry (ab|cd)"""
  ab_idx, counts_ab = get_2c_combs(n_orbs)

  # block idx of (ab|cd)
  ab_block_idx = np.vstack(np.triu_indices(len(ab_idx))).T
  offdiag_ab_block = ab_block_idx[:, 0] != ab_block_idx[:, 1]
  counts_ab_block = offdiag_ab_block + np.ones(len(ab_block_idx))
  in_block_counts = (
    counts_ab[ab_block_idx[:, 0]] * counts_ab[ab_block_idx[:, 1]]
  )
  between_block_counts = counts_ab_block

  counts_abcd = in_block_counts * between_block_counts
  counts_abcd = counts_abcd.astype(np.int32)

  abcd_idx = np.hstack([ab_idx[ab_block_idx[:, 0]], ab_idx[ab_block_idx[:, 1]]])
  return abcd_idx, counts_abcd


def calc_hamiltonian(mol: pyscf.gto.mole.Mole, cfg: DFTConfig) -> Hamiltonian:
  """Discretize the electron Hamiltonian in GTO basis (except for XC)."""
  ovlp = mol.intor('int1e_ovlp_sph')
  # TODO: refactor this part
  kin = mol.intor_symmetric('int1e_kin')
  ext = mol.intor_symmetric('int1e_nuc')
  eri = 0.5 * mol.intor('int2e')

  n_mos = len(ovlp)
  mo_ab_idx, mo_counts_ab = get_2c_combs(n_mos)
  mo_abcd_idx, mo_counts_abcd = get_4c_combs(n_mos)

  # reduce symmetry
  kin = kin[mo_ab_idx[:, 0], mo_ab_idx[:, 1]]
  ext = ext[mo_ab_idx[:, 0], mo_ab_idx[:, 1]]
  eri = eri[mo_abcd_idx[:, 0], mo_abcd_idx[:, 1], mo_abcd_idx[:, 2],
            mo_abcd_idx[:, 3]]

  nocc = get_occupation_mask(mol, mol.spin)

  def mo_coeff_fn(params):
    """get GTO representation of MO"""
    mo_coeff = qr_factor_param(params, ovlp)
    if cfg.rks:  # restrictied mo
      mo_coeff_spin = jnp.repeat(mo_coeff[None], 2, 0)  # add spin axis
    else:
      mo_coeff_spin = mo_coeff
    mo_coeff_spin *= nocc[:, :, None]  # apply spin mask
    mo_coeff_spin = mo_coeff_spin.reshape(-1, n_mos)  # flatten
    return mo_coeff_spin

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


def euclidean_distance(x, y):
  """Euclidean distance."""
  return jnp.sqrt(jnp.sum((x - y)**2 + 1e-18))


def distmat(x, y=None):
  """Distance matrix."""
  if y is None:
    y = x
  return jax.vmap(
    lambda x1: jax.vmap(lambda y1: euclidean_distance(x1, y1))(y)
  )(
    x
  )


def set_diag_zero(x):
  """Set diagonal items to zero."""
  return x.at[jnp.diag_indices(x.shape[0])].set(0)


def e_nuclear(nuclei):
  """
    Potential energy between atomic nuclears.
  """
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  dist_nuc = distmat(nuclei_loc)
  charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
  charge_outer = set_diag_zero(charge_outer)
  return jnp.sum(charge_outer / (dist_nuc + 1e-15)) / 2


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
