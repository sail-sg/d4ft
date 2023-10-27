# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jaxtyping import Array, Float

from d4ft.config import SCFConfig
from d4ft.hamiltonian.ortho import sqrt_inv
from d4ft.integral.gto.cgto import CGTO
from d4ft.logger import RunLogger
from d4ft.types import MoCoeffFlat


def scf(
  cfg: SCFConfig,
  cgto: CGTO,
  mo_coeff: MoCoeffFlat,
  ovlp: Float[Array, "2 nao nao"],
  fock_fn: Callable,
  energy_fn: Callable,
  restricted: bool,
) -> RunLogger:

  ovlp_sqrt_inv = sqrt_inv(ovlp)

  transpose_axis = (1, 0) if restricted else (0, 2, 1)

  @jax.jit
  def scf_iter(mo_coeff, fock):
    new_fock = fock_fn(mo_coeff)
    fock = (1 - cfg.momentum) * new_fock + cfg.momentum * fock
    fock_ortho = ovlp_sqrt_inv @ fock @ ovlp_sqrt_inv.T
    e_orb, mo_coeff_ortho = jnp.linalg.eigh(fock_ortho)
    mo_coeff = ovlp_sqrt_inv.T @ mo_coeff_ortho
    mo_coeff = jnp.transpose(mo_coeff, transpose_axis)
    return e_orb, mo_coeff, fock

  fock = jnp.eye(cgto.nao)  # initial guess
  logger = RunLogger()
  converged = False
  e_total_tm1 = 0.0
  for step in range(cfg.epochs):
    e_orb, mo_coeff, new_fock = scf_iter(mo_coeff, fock)
    fock = new_fock
    logging.info(f"{e_orb=}")
    residual = jnp.eye(cgto.nao) - mo_coeff[0] @ ovlp @ mo_coeff[0].T
    max_res = np.abs(residual).max()
    _, (energies, _) = energy_fn(
      jnp.transpose(mo_coeff, transpose_axis) * cgto.nocc[:, :, None]
    )
    e_delta = np.abs(energies.e_total - e_total_tm1)
    thresh = max(max_res, e_delta)
    logger.log_step(energies, step, thresh)
    logger.get_segment_summary()

    e_total_tm1 = energies.e_total

    if step > 1 and thresh < cfg.converge_threshold:
      converged = True
      break

  logging.info(f"Converged: {converged}")

  return logger
