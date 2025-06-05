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

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from d4ft.hamiltonian.cgto_intors import get_cgto_intor
from d4ft.hamiltonian.nuclear import e_nuclear
from d4ft.hamiltonian.ortho import sqrt_inv
from d4ft.integral.gto.cgto import CGTO
from d4ft.types import (
  Aux,
  CGTOSymTensorFns,
  CGTOSymTensorIncore,
  Energies,
  Hamiltonian,
  MoCoeff,
  MoCoeffFlat,
)


# TODO: add more API to H
# 1. params -> cgto_tensor_fns -> cgto_e_tensors
# 2. params,  -> e_fn -> cgto_e_tensors
def mf_cgto(
  cgto: CGTO,
  cgto_tensor_fns: CGTOSymTensorFns,
  mo_coeff_fn: Callable,
  xc_fn: Optional[Callable] = None,
) -> Tuple[Callable, Hamiltonian]:
  r"""Mean-field level calculation with CGTO, i.e. electron Hamiltonian
  with single Slater determinant ansatz / Hartree-Fock, discretized the
  in CGTO/AO basis.

  `xc_fn` is function that calculates the exact exchange energy. To perform
  Kohn-Sham DFT calculation simply replace it with a XC functional.
  All energy integral are computed analytically except for XC which is
  integrated numerically with quadrature.

  It compose mo_coeff_fn with the cgto intors, and create a energy_fn
  that computes the total energy with logging.
  """
  cgto_intors = get_cgto_intor(cgto, "obsa")
  e_tensor_fn = lambda: cgto_tensor_fns.get_incore_tensors(cgto)

  def nuc_fn() -> Float[Array, ""]:
    return e_nuclear(jnp.array(cgto.atom_coords), jnp.array(cgto.charge))

  def energy_fn(
    cgto_e_tensors: Optional[CGTOSymTensorIncore] = None,
    mo_coeff: MoCoeffFlat = None
  ) -> Tuple[Float[Array, ""], Aux]:
    """
    if cgto_e_tensors is not None, perform incore calculation, i.e.
    use precomputed 2c/4c integrals
    """
    if cgto_e_tensors is None:  # on-the-fly calculation
      cgto_e_tensors = e_tensor_fn()

    ovlp = cgto_intors.ovlp_fn(cgto_e_tensors)

    if mo_coeff is None:
      mo_coeff = mo_coeff_fn(ovlp_sqrt_inv=sqrt_inv(ovlp))

    mo_energies = [e_fn(mo_coeff, cgto_e_tensors) for e_fn in cgto_intors[1:]]
    e_kin, e_ext, e_har, e_exc = mo_energies

    if xc_fn is not None:
      # calculate the exchange-correlation energy using XC functional
      e_xc = xc_fn(mo_coeff)

    else:  # use the exact exchange energy
      e_xc = e_exc

    e_nuc = nuc_fn()
    e_total = e_kin + e_ext + e_har + e_xc + e_nuc
    energies = Energies(e_total, e_kin, e_ext, e_har, e_xc, e_nuc)
    loss = e_total
    return loss, energies

  def _xc_fn(
    cgto_e_tensors: Optional[CGTOSymTensorIncore] = None
  ) -> Float[Array, ""]:
    """Get the XC functional, if any."""
    if cgto_e_tensors is None:  # on-the-fly calculation
      cgto_e_tensors = e_tensor_fn()
    ovlp = cgto_intors.ovlp_fn(cgto_e_tensors)
    mo_coeff = mo_coeff_fn(ovlp_sqrt_inv=sqrt_inv(ovlp))
    return xc_fn(mo_coeff)

  hamiltonian = Hamiltonian(
    cgto_intors,
    nuc_fn,
    energy_fn,
    mo_coeff_fn,
    pgto_fn=lambda: cgto.pgto,
    coeff_fn=lambda: cgto.coeff,
    e_tensor_fn=e_tensor_fn,
    xc_fn=_xc_fn,
  )

  return energy_fn, hamiltonian
