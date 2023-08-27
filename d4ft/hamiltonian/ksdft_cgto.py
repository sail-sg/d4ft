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


def ksdft_cgto(
  cgto: CGTO,
  cgto_intors: CGTOIntors,
  xc_fn: Callable,
  mo_coeff_fn: Optional[Callable[[], MoCoeffFlat]] = None,
  ret_mo_grads: bool = False,
) -> Tuple[Callable, Hamiltonian]:
  """Electron Hamiltonian with single Slater determinant ansatz (Hartree-Fock),
  discretized the in CGTO/AO basis. All energy integral are computed
  analytically except for XC which is integrated numerically with quadrature.

  It compose mo_coeff_fn with the cgto intors, and create a energy_fn
  that computes the total energy with logging.
  """

  kin_fn, ext_fn, har_fn = cgto_intors

  def nuc_fn() -> Float[Array, ""]:
    return e_nuclear(jnp.array(cgto.atom_coords), jnp.array(cgto.charge))

  e_fns = [kin_fn, ext_fn, har_fn, xc_fn]

  def energy_fn(mo_coeff: MoCoeffFlat) -> Tuple[Float[Array, ""], Aux]:
    if ret_mo_grads:
      val_and_grads = [jax.value_and_grad(e_fn)(mo_coeff) for e_fn in e_fns]
      mo_energies, mo_grads = zip(*val_and_grads)
      kin_grads, ext_grads, har_grads, xc_grads = mo_grads
      grads = Grads(kin_grads, ext_grads, har_grads, xc_grads)
    else:
      mo_energies = [e_fn(mo_coeff) for e_fn in e_fns]
      grads = None
    e_kin, e_ext, e_har, e_xc = mo_energies
    e_nuc = nuc_fn()
    e_total = sum(mo_energies) + e_nuc
    energies = Energies(e_total, e_kin, e_ext, e_har, e_xc, e_nuc)
    return e_total, (energies, grads)

  if mo_coeff_fn is None:
    return energy_fn, Hamiltonian(
      kin_fn, ext_fn, har_fn, xc_fn, nuc_fn, energy_fn, mo_coeff_fn
    )

  kin_fn_, ext_fn_, har_fn_, xc_fn_, energy_fn_ = [
    compose(e_fn, mo_coeff_fn)
    for e_fn in [kin_fn, ext_fn, har_fn, xc_fn, energy_fn]
  ]

  hamiltonian = Hamiltonian(
    kin_fn_, ext_fn_, har_fn_, xc_fn_, nuc_fn, energy_fn_, mo_coeff_fn
  )

  return energy_fn_, hamiltonian
