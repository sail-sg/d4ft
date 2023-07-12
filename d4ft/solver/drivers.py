#!/usr/bin/env python3
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
"""High level routine for full calculations"""

from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax_xc
from absl import logging
from d4ft.config import D4FTConfig
from d4ft.hamiltonian.cgto_intors import get_cgto_fock_fn, get_cgto_intor
from d4ft.hamiltonian.dft_cgto import dft_cgto
from d4ft.hamiltonian.ortho import qr_factor, sqrt_inv
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import incore_int_sym
from d4ft.integral.quadrature.grids import DifferentiableGrids
from d4ft.logger import RunLogger
from d4ft.solver.pyscf_wrapper import pyscf_wrapper
from d4ft.solver.sgd import sgd
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.types import Hamiltonian
from d4ft.utils import make_constant_fn
from d4ft.xc import get_xc_intor
from jaxtyping import Array, Float


def incore_hf_cgto(cfg: D4FTConfig):
  pyscf_mol = get_pyscf_mol(
    cfg.mol_cfg.mol, cfg.mol_cfg.basis, cfg.mol_cfg.spin, cfg.mol_cfg.charge,
    cfg.mol_cfg.geometry_source
  )
  mol = Mol.from_pyscf_mol(pyscf_mol)
  cfg.validate(mol.spin, mol.charge)
  cgto = CGTO.from_mol(mol)

  # TODO: intor.split() for pmap / batched
  s2 = obsa.angular_static_args(*[cgto.primitives.angular] * 2)
  s4 = obsa.angular_static_args(*[cgto.primitives.angular] * 4)
  incore_energy_tensors = incore_int_sym(cgto, s2, s4)
  return incore_energy_tensors, pyscf_mol, cgto


def incore_cgto_scf_dft(cfg: D4FTConfig) -> None:
  """Solve for ground state of a molecular system with SCF KS-DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore.

  NOTE: since jax-xc doesn't have vxc yet the vxc here is fixed to LDA
  """
  key = jax.random.PRNGKey(cfg.optim_cfg.rng_seed)
  incore_energy_tensors, pyscf_mol, cgto = incore_hf_cgto(cfg)

  dg = DifferentiableGrids(pyscf_mol)
  dg.level = cfg.direct_min_cfg.quad_level
  # TODO: test geometry optimization
  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  cgto_fock_fn = get_cgto_fock_fn(
    cgto,
    incore_energy_tensors,
    grids_and_weights,
    polarized=not cfg.direct_min_cfg.rks
  )
  cgto_fock_jit = jax.jit(cgto_fock_fn)

  # get initial mo_coeff
  mo_coeff_fn = partial(cgto.get_mo_coeff, rks=cfg.direct_min_cfg.rks)
  mo_coeff_fn = hk.without_apply_rng(hk.transform(mo_coeff_fn))
  params = mo_coeff_fn.init(key)
  mo_coeff = mo_coeff_fn.apply(params)

  fock = cgto_fock_jit(mo_coeff)
  _, new_mo = jnp.linalg.eigh(fock)
  breakpoint()


def incore_cgto_direct_opt_dft(cfg: D4FTConfig) -> None:
  """Solve for ground state of a molecular system with direct optimization DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore."""
  key = jax.random.PRNGKey(cfg.optim_cfg.rng_seed)
  xc_functional = getattr(jax_xc, cfg.direct_min_cfg.xc_type)

  incore_energy_tensors, pyscf_mol, cgto = incore_hf_cgto(cfg)

  dg = DifferentiableGrids(pyscf_mol)
  dg.level = cfg.direct_min_cfg.quad_level
  # TODO: test geometry optimization
  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  # TODO: change this to use obsa
  ovlp: Float[Array, "a a"] = pyscf_mol.intor('int1e_ovlp_sph')

  def H_factory() -> Tuple[Callable, Hamiltonian]:
    # TODO: out-of-core + basis optimization
    # cgto_hk = cgto.to_hk()
    cgto_hk = cgto
    cgto_intor = get_cgto_intor(
      cgto_hk, intor="obsa", incore_energy_tensors=incore_energy_tensors
    )
    mo_coeff_fn = partial(
      cgto_hk.get_mo_coeff,
      rks=cfg.direct_min_cfg.rks,
      ortho_fn=qr_factor,
      ovlp_sqrt_inv=sqrt_inv(ovlp),
    )
    xc_fn = get_xc_intor(
      grids_and_weights,
      cgto_hk,
      xc_functional,
      polarized=not cfg.direct_min_cfg.rks
    )
    return dft_cgto(cgto_hk, cgto_intor, xc_fn, mo_coeff_fn)

  e_total, traj, H = sgd(cfg.direct_min_cfg, cfg.optim_cfg, H_factory, key)
  lowest_e = jnp.stack([t.energies.e_total for t in traj]).min()
  logging.info(f"lowest total energy: {lowest_e}")
  # breakpoint()

  # ovlp_sqrt_inv = sqrt_inv(ovlp)
  # breakpoint()

  # rdm1 = get_rdm1(traj[-1].mo_coeff)
  # scf_mo_coeff = pyscf_wrapper(
  #   pyscf_mol,
  #   cfg.direct_min_cfg.rks,
  #   cfg.direct_min_cfg.xc_type,
  #   cfg.direct_min_cfg.quad_level,
  #   # rdm1=rdm1,
  #   rdm1=traj[-1].mo_coeff,
  # )
  # breakpoint()


def incore_cgto_pyscf_dft_benchmark(cfg: D4FTConfig) -> None:
  """Call PySCF to solve for ground state of a molecular system with SCF DFT,
  then load the computed MO coefficients from PySCF and redo the energy integral
  with obsa, where the energy tensors are precomputed/incore."""
  incore_energy_tensors, pyscf_mol, cgto = incore_hf_cgto(cfg)

  dg = DifferentiableGrids(pyscf_mol)
  dg.level = cfg.direct_min_cfg.quad_level
  # TODO: test geometry optimization
  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  cgto_intor = get_cgto_intor(
    cgto, intor="obsa", incore_energy_tensors=incore_energy_tensors
  )
  xc_functional = getattr(jax_xc, cfg.direct_min_cfg.xc_type)
  xc_fn = get_xc_intor(
    grids_and_weights,
    cgto,
    xc_functional,
    polarized=not cfg.direct_min_cfg.rks
  )

  # solve for ground state with PySCF and get the mo_coeff
  mo_coeff = pyscf_wrapper(
    pyscf_mol, cfg.direct_min_cfg.rks, cfg.direct_min_cfg.xc_type,
    cfg.direct_min_cfg.quad_level
  )
  breakpoint()

  # add spin and apply occupation mask
  mo_coeff *= cgto.nocc[:, :, None]

  # eval with d4ft
  _, H = dft_cgto(
    cgto, cgto_intor, xc_fn, mo_coeff_fn=make_constant_fn(mo_coeff)
  )

  _, (energies, _) = H.energy_fn(mo_coeff)

  logger = RunLogger()
  logger.log_step(energies, 0)
  logger.get_segment_summary()
  logging.info(f"1e energy:{energies.e_kin + energies.e_ext}")
