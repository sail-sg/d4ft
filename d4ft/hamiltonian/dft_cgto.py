from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from d4ft.hamiltonian.nuclear import e_nuclear
from d4ft.integral.gto.cgto import CGTO
from d4ft.types import (
  Aux, CGTOIntors, Energies, Grads, Hamiltonian, MoCoeffFlat
)
from d4ft.utils import compose
from jaxtyping import Array, Float


def dft_cgto(
  cgto: CGTO, cgto_intors: CGTOIntors, xc_fn: Callable,
  mo_coeff_fn: Callable[[], MoCoeffFlat]
) -> Tuple[Callable, Hamiltonian]:
  """Electron Hamiltonian discretized the in GTO basis,
  except for XC which is integrated numerically with quadrature."""

  kin_fn, ext_fn, eri_fn = cgto_intors

  def nuc_fn() -> Float[Array, ""]:
    return e_nuclear(jnp.array(cgto.atom_coords), jnp.array(cgto.charge))

  def energy_fn(mo_coeff: MoCoeffFlat) -> Tuple[Float[Array, ""], Aux]:
    val_and_grads = [
      jax.value_and_grad(e_fn)(mo_coeff)
      for e_fn in [kin_fn, ext_fn, eri_fn, xc_fn]
    ]
    mo_energies, mo_grads = zip(*val_and_grads)
    e_kin, e_ext, e_eri, e_xc = mo_energies
    kin_grads, ext_grads, eri_grads, xc_grads = mo_grads
    e_nuc = nuc_fn()
    e_total = sum(mo_energies) + e_nuc
    energies = Energies(e_total, e_kin, e_ext, e_eri, e_xc, e_nuc)
    grads = Grads(kin_grads, ext_grads, eri_grads, xc_grads)
    return e_total, (energies, grads)

  kin_fn_, ext_fn_, eri_fn_, xc_fn_, energy_fn_ = [
    compose(e_fn, mo_coeff_fn)
    for e_fn in [kin_fn, ext_fn, eri_fn, xc_fn, energy_fn]
  ]

  hamiltonian = Hamiltonian(
    kin_fn_, ext_fn_, eri_fn_, xc_fn_, nuc_fn, energy_fn_, mo_coeff_fn
  )

  return energy_fn_, hamiltonian
