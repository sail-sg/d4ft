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

from d4ft.hamiltonian.ortho import sqrt_inv
from d4ft.hamiltonian.cgto_intors import get_cgto_intor
from d4ft.hamiltonian.nuclear import e_nuclear
from d4ft.integral.gto.cgto import CGTO
from d4ft.types import (
  Aux,
  CGTOIntors,
  Energies,
  Grads,
  Hamiltonian,
  MoCoeffFlat,
)
from d4ft.utils import compose


# TODO: add more API to H
# 1. params -> cgto_tensor_fns -> cgto_e_tensors
# 2. params,  -> e_fn -> cgto_e_tensors
def mf_cgto(
  cgto: CGTO,
  cgto_tensor_fns: Callable,
  # cgto_intors: CGTOIntors,
  mo_coeff_fn: Callable[[], MoCoeffFlat],
  xc_fn: Optional[Callable] = None,
  vxc_fn: Optional[Callable] = None,
  ret_mo_grads: bool = False,
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

  def nuc_fn() -> Float[Array, ""]:
    return e_nuclear(jnp.array(cgto.atom_coords), jnp.array(cgto.charge))

  def energy_fn(
    cgto_e_tensors: Optional[CGTOSymTensorIncore] = None
  ) -> Tuple[Float[Array, ""], Aux]:
    """
    if cgto_e_tensors is not None, perform incore calculation, i.e.
    use precomputed 2c/4c integrals
    """
    if cgto_e_tensors is not None: # incore
      cgto_intor = get_cgto_intor(cgto, cgto_e_tensors=cgto_e_tensors)
    else: # on-the-fly
      cgto_intor = get_cgto_intor(cgto, cgto_tensor_fns=cgto_tensor_fns)

    mo_coeff = mo_coeff_fn(ovlp_sqrt_inv=sqrt_inv(cgto_intor.ovlp_fn()))
    mo_energies = [e_fn(mo_coeff) for e_fn in cgto_intors[1:]]
    grads = None
    e_kin, e_ext, e_har = mo_energies
    e_xc = xc_fn(mo_coeff)

    e_nuc = nuc_fn()
    e_total = sum(mo_energies) + e_nuc
    energies = Energies(e_total, e_kin, e_ext, e_har, e_xc, e_nuc)
    if vxc_fn is None:
      loss = e_total
    else:
      vxc_loss = vxc_fn(mo_coeff)**2
      loss = e_total + vxc_loss
    return loss, (energies, grads)

  if mo_coeff_fn is None:
    return energy_fn, Hamiltonian(cgto_intors, nuc_fn, energy_fn, mo_coeff_fn)

  # cgto_intors_ = CGTOIntors(
  #   *[compose(e_fn, mo_coeff_fn) for e_fn in cgto_intors]
  # )
  energy_fn_ = compose(energy_fn, mo_coeff_fn)
  hamiltonian = Hamiltonian(
    cgto_intors_,
    nuc_fn,
    energy_fn_,
    mo_coeff_fn,
    pgto_fn=lambda: cgto.pgto,
    coeff_fn=lambda: cgto.coeff
  )
  return energy_fn_, hamiltonian
