import haiku as hk
import jax.numpy as jnp
import pyscf
from jaxtyping import Array, Float

from d4ft.config import DFTConfig
from d4ft.integral.gto import symmetry
from d4ft.integral.gto.lcgto import LCGTO
from d4ft.integral.obara_saika.driver import incore_int
from d4ft.mo_coeff_fn import get_mo_coeff_fn
from d4ft.nuclear import e_nuclear
from d4ft.types import (
  ETensorsIncore, Hamiltonian, IdxCount2C, IdxCount4C, MoCoeff
)
from d4ft.xc import get_xc_intor


def libcint_incore(
  mol: pyscf.gto.mole.Mole,
  mo_ab_idx_counts: IdxCount2C,
  mo_abcd_idx_counts: IdxCount4C,
) -> ETensorsIncore:
  """Get tensor incore using libcint, then reduce symmetry."""
  kin = mol.intor_symmetric('int1e_kin')
  ext = mol.intor_symmetric('int1e_nuc')
  eri = 0.5 * mol.intor('int2e')
  kin = kin[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
  ext = ext[mo_ab_idx_counts[:, 0], mo_ab_idx_counts[:, 1]]
  eri = eri[mo_abcd_idx_counts[:, 0], mo_abcd_idx_counts[:, 1],
            mo_abcd_idx_counts[:, 2], mo_abcd_idx_counts[:, 3]]
  counts_ab = mo_ab_idx_counts[:, 2]
  counts_abcd = mo_abcd_idx_counts[:, 4]
  kin *= counts_ab
  ext *= counts_ab
  eri *= counts_abcd
  return kin, ext, eri


def calc_hamiltonian(mol: pyscf.gto.mole.Mole, cfg: DFTConfig) -> Hamiltonian:
  """Discretize the electron Hamiltonian in GTO basis (except for XC)."""
  get_lcgto = LCGTO.from_pyscf_mol(mol, use_hk=True)
  get_lcgto = hk.without_apply_rng(hk.transform(get_lcgto))
  gparams = get_lcgto.init(1)
  gtos = get_lcgto.apply(gparams)
  nmo = len(gtos.cgto_splits)

  mo_ab_idx_counts = symmetry.get_2c_sym_idx(nmo)
  mo_abcd_idx_counts = symmetry.get_4c_sym_idx(nmo)

  if cfg.incore:
    if cfg.intor == "obsa":
      kin, ext, eri = incore_int(gtos)
    elif cfg.intor == "libcint":
      kin, ext, eri = libcint_incore(mol, mo_ab_idx_counts, mo_abcd_idx_counts)

    def kin_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = mo_coeff.T @ mo_coeff
      rdm1_2c_ab = rdm1[hamiltonian.mo_ab_idx_counts[:, 0],
                        hamiltonian.mo_ab_idx_counts[:, 1]]
      e_kin = jnp.sum(kin * rdm1_2c_ab)
      return e_kin

    def ext_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = mo_coeff.T @ mo_coeff
      rdm1_2c_ab = rdm1[hamiltonian.mo_ab_idx_counts[:, 0],
                        hamiltonian.mo_ab_idx_counts[:, 1]]
      e_ext = jnp.sum(ext * rdm1_2c_ab)
      return e_ext

    def eri_fn(mo_coeff: MoCoeff) -> Float[Array, ""]:
      rdm1 = mo_coeff.T @ mo_coeff
      rdm1_4c_ab = rdm1[hamiltonian.mo_abcd_idx_counts[:, 0],
                        hamiltonian.mo_abcd_idx_counts[:, 1]]
      rdm1_4c_cd = rdm1[hamiltonian.mo_abcd_idx_counts[:, 2],
                        hamiltonian.mo_abcd_idx_counts[:, 3]]
      e_eri = jnp.sum(eri * rdm1_4c_ab * rdm1_4c_cd)
      return e_eri

    xc_intor = get_xc_intor(mol, cfg.xc_type, cfg.quad_level)

    def xc_fn(mo_coeff) -> float:
      e_xc = xc_intor(mo_coeff)
      return e_xc

  else:
    pass

  hamiltonian = Hamiltonian(
    kin_fn,
    ext_fn,
    eri_fn,
    xc_fn,
    mo_ab_idx_counts,
    mo_abcd_idx_counts,
    get_mo_coeff_fn(mol, cfg.rks),
    e_nuclear(jnp.array(mol.atom_coords()), jnp.array(mol.atom_charges())),
  )

  return hamiltonian
