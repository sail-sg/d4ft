"""MO coefficient parametrization and occupation processing."""

from typing import Union

import jax.numpy as jnp
import pyscf
from jaxtyping import Array, Float


def sqrt_root_inv(mat: Float[Array, "a a"]) -> Float[Array, "a a"]:
  """Square root of inverse."""
  v, u = jnp.linalg.eigh(mat)
  v = jnp.clip(v, a_min=0)
  v = jnp.diag(jnp.real(v)**(-1 / 2))
  ut = jnp.real(u).transpose()
  return jnp.matmul(v, ut)


def qr_factor_param(
  params: Union[Float[Array, "nao nao"], Float[Array, "2 nao nao"]],
  ovlp: Float[Array, "nao nao"],
  rks: bool = True,
) -> Union[Float[Array, "nao nao"], Float[Array, "2 nao nao"]]:
  """Parametrize the generalized Stiefel manifold (CS^{-1/2}C=I)
  with qr factor"""
  orthogonal, _ = jnp.linalg.qr(params)
  transpose_axis = (1, 0) if rks else (0, 2, 1)
  orthogonal = jnp.transpose(orthogonal, transpose_axis)
  mo_coeff = orthogonal @ sqrt_root_inv(ovlp)
  return mo_coeff


def get_occupation_mask(mol: pyscf.gto.mole.Mole):
  tot_electron = mol.tot_electrons()
  nao = mol.nao  # number of atomic orbitals
  nocc = jnp.zeros([2, nao])  # number of occupied orbital.
  nmo_up = (tot_electron + mol.spin) // 2
  nmo_dn = (tot_electron - mol.spin) // 2
  nocc = nocc.at[0, :nmo_up].set(1)
  nocc = nocc.at[1, :nmo_dn].set(1)
  return nocc


def get_mo_coeff_fn(mol: pyscf.gto.mole.Mole, rks: bool):
  """NOTE: Currently only support QR parameterization"""
  ovlp: Float[Array, "nao nao"] = mol.intor('int1e_ovlp_sph')
  nocc = get_occupation_mask(mol)
  nmo = len(ovlp)

  def mo_coeff_fn(
    params: Union[Float[Array, "nao nao"], Float[Array, "2 nao nao"]]
  ):
    """get GTO representation of MO"""
    mo_coeff = qr_factor_param(params, ovlp)
    if rks:  # restrictied mo
      mo_coeff_spin = jnp.repeat(mo_coeff[None], 2, 0)  # add spin axis
    else:
      mo_coeff_spin = mo_coeff
    mo_coeff_spin *= nocc[:, :, None]  # apply spin mask
    mo_coeff_spin = mo_coeff_spin.reshape(-1, nmo)  # flatten
    return mo_coeff_spin

  return mo_coeff_fn
