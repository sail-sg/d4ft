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
from absl import logging
from jaxtyping import Array, Float

from d4ft.config import D4FTConfig
from d4ft.hamiltonian.cgto_intors import get_cgto_fock_fn, get_cgto_intor
from d4ft.hamiltonian.dft_cgto import dft_cgto
from d4ft.hamiltonian.nuclear import e_nuclear
from d4ft.hamiltonian.ortho import qr_factor, sqrt_inv
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import incore_int_sym
from d4ft.integral.quadrature.grids import DifferentiableGrids
from d4ft.logger import RunLogger
from d4ft.solver.pyscf_wrapper import pyscf_wrapper
from d4ft.solver.sgd import sgd
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.types import Energies, Hamiltonian
from d4ft.utils import make_constant_fn
from d4ft.xc import get_lda_vxc, get_xc_functional, get_xc_intor


def incore_hf_cgto(cfg: D4FTConfig):
  pyscf_mol = get_pyscf_mol(
    cfg.mol_cfg.mol, cfg.mol_cfg.basis, cfg.mol_cfg.spin, cfg.mol_cfg.charge,
    cfg.mol_cfg.geometry_source
  )
  mol = Mol.from_pyscf_mol(pyscf_mol)
  cfg.validate(mol.spin, mol.charge)
  cgto_cart= CGTO.from_mol(mol)
  cgto = CGTO.from_cart(cgto_cart)

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
  key = jax.random.PRNGKey(cfg.dft_cfg.rng_seed)
  incore_energy_tensors, pyscf_mol, cgto = incore_hf_cgto(cfg)

  dg = DifferentiableGrids(pyscf_mol)
  dg.level = cfg.intor_cfg.quad_level
  # TODO: test geometry optimization
  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  vxc_fn = get_lda_vxc(grids_and_weights, cgto, polarized=not cfg.dft_cfg.rks)
  cgto_fock_fn = get_cgto_fock_fn(cgto, incore_energy_tensors, vxc_fn)
  cgto_fock_jit = jax.jit(cgto_fock_fn)

  # get initial mo_coeff
  mo_coeff_fn = partial(cgto.get_mo_coeff, rks=cfg.dft_cfg.rks)
  mo_coeff_fn = hk.without_apply_rng(hk.transform(mo_coeff_fn))
  params = mo_coeff_fn.init(key)
  mo_coeff = mo_coeff_fn.apply(params, apply_spin_mask=False)

  polarized = not cfg.dft_cfg.rks
  xc_func = get_xc_functional(cfg.dft_cfg.xc_type, polarized)
  xc_fn = get_xc_intor(grids_and_weights, cgto, xc_func, polarized)
  kin_fn, ext_fn, har_fn = get_cgto_intor(
    cgto, intor="obsa", incore_energy_tensors=incore_energy_tensors
  )
  e_nuc = e_nuclear(jnp.array(cgto.atom_coords), jnp.array(cgto.charge))

  @jax.jit
  def energy_fn(mo_coeff):
    e_kin = kin_fn(mo_coeff)
    e_ext = ext_fn(mo_coeff)
    e_har = har_fn(mo_coeff)
    e_xc = xc_fn(mo_coeff)
    e_total = e_kin + e_ext + e_har + e_xc + e_nuc
    energies = Energies(e_total, e_kin, e_ext, e_har, e_xc, e_nuc)
    return energies

  transpose_axis = (1, 0) if cfg.dft_cfg.rks else (0, 2, 1)

  @jax.jit
  def update(mo_coeff, fock):
    _, mo_coeff = jnp.linalg.eigh(fock)
    mo_coeff = jnp.transpose(mo_coeff, transpose_axis)
    return mo_coeff

  fock = jnp.eye(cgto.nao)  # initial guess
  logger = RunLogger()
  for step in range(cfg.scf_cfg.epochs):
    new_fock = cgto_fock_jit(mo_coeff)
    fock = (1 - cfg.scf_cfg.momentum) * new_fock + cfg.scf_cfg.momentum * fock
    mo_coeff = update(mo_coeff, fock)
    energies = energy_fn(mo_coeff * cgto.nocc[:, :, None])
    logger.log_step(energies, step)
    logger.get_segment_summary()


def incore_cgto_direct_opt_dft(cfg: D4FTConfig) -> float:
  """Solve for ground state of a molecular system with direct optimization DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore."""
  key = jax.random.PRNGKey(cfg.dft_cfg.rng_seed)

  polarized = not cfg.dft_cfg.rks
  xc_func = get_xc_functional(cfg.dft_cfg.xc_type, polarized)

  incore_energy_tensors, pyscf_mol, cgto = incore_hf_cgto(cfg)

  dg = DifferentiableGrids(pyscf_mol)
  dg.level = cfg.intor_cfg.quad_level
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
      rks=cfg.dft_cfg.rks,
      ortho_fn=qr_factor,
      ovlp_sqrt_inv=sqrt_inv(ovlp),
    )
    xc_fn = get_xc_intor(grids_and_weights, cgto_hk, xc_func, polarized)
    return dft_cgto(cgto_hk, cgto_intor, xc_fn, mo_coeff_fn)

  # e_total = scipy_opt(cfg.gd_cfg, H_factory, key)
  # breakpoint()

  logger, traj, H = sgd(cfg.gd_cfg, H_factory, key)
  min_e_step = logger.data_df.e_total.astype(float).idxmin()
  logging.info(f"lowest total energy: \n {logger.data_df.iloc[min_e_step]}")
  lowest_e = logger.data_df.e_total.astype(float).min()
  return lowest_e

  # NOTE: diagonalize the fock matrix gives a different mo_coeff
  # rdm1 = get_rdm1(traj[-1].mo_coeff)
  # scf_mo_coeff = pyscf_wrapper(
  #   pyscf_mol,
  #   cfg.dft_cfg.rks,
  #   cfg.dft_cfg.xc_type,
  #   cfg.intor_cfg.quad_level,
  #   rdm1=rdm1,
  # )
  # breakpoint()


def incore_cgto_pyscf_dft_benchmark(cfg: D4FTConfig) -> None:
  """Call PySCF to solve for ground state of a molecular system with SCF DFT,
  then load the computed MO coefficients from PySCF and redo the energy integral
  with obsa, where the energy tensors are precomputed/incore."""
  incore_energy_tensors, pyscf_mol, cgto = incore_hf_cgto(cfg)

  dg = DifferentiableGrids(pyscf_mol)
  dg.level = cfg.intor_cfg.quad_level
  # TODO: test geometry optimization
  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  cgto_intor = get_cgto_intor(
    cgto, intor="obsa", incore_energy_tensors=incore_energy_tensors
  )
  polarized = not cfg.dft_cfg.rks
  xc_func = get_xc_functional(cfg.dft_cfg.xc_type, polarized)
  xc_fn = get_xc_intor(grids_and_weights, cgto, xc_func, polarized)

  # solve for ground state with PySCF and get the mo_coeff
  mo_coeff = pyscf_wrapper(
    pyscf_mol, cfg.dft_cfg.rks, cfg.dft_cfg.xc_type, cfg.intor_cfg.quad_level
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
