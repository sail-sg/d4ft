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

import pickle
from functools import partial
from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from absl import logging

from d4ft.config import D4FTConfig
from d4ft.hamiltonian.cgto_intors import (
  get_cgto_fock_fn,
  get_cgto_intor,
  get_ovlp,
)
from d4ft.hamiltonian.mf_cgto import mf_cgto
from d4ft.hamiltonian.nuclear import e_nuclear
from d4ft.hamiltonian.ortho import qr_factor, sqrt_inv
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import (
  CGTOSymTensorFns,
  get_cgto_sym_tensor_fns,
)
from d4ft.integral.quadrature.grids import DifferentiableGrids
from d4ft.logger import RunLogger
from d4ft.solver.pyscf_wrapper import pyscf_wrapper
from d4ft.solver.sgd import sgd
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.types import Energies, Hamiltonian
from d4ft.utils import make_constant_fn
from d4ft.xc import get_lda_vxc, get_xc_functional, get_xc_intor


def build_mf_cgto(
  cfg: D4FTConfig
) -> Tuple[
  CGTOSymTensorFns,
  pyscf.gto.mole.Mole,
  CGTO,
  Optional[DifferentiableGrids],
]:
  """Build the CGTO basis with in-core intor for the mean-field calculations
  (i.e. HF and KS-DFT). For KS-DFT we also need to build the grids for the
  numerical integration of the XC functional"""
  pyscf_mol = get_pyscf_mol(
    cfg.sys_cfg.mol, cfg.sys_cfg.basis, cfg.sys_cfg.spin, cfg.sys_cfg.charge,
    cfg.sys_cfg.geometry_source
  )
  mol = Mol.from_pyscf_mol(pyscf_mol)
  cfg.validate(mol.spin, mol.charge)
  cgto = CGTO.from_mol(mol)

  # TODO: intor.split() for pmap / batched
  s2 = obsa.angular_static_args(*[cgto.pgto.angular] * 2)
  s4 = obsa.angular_static_args(*[cgto.pgto.angular] * 4)
  cgto_tensor_fns = get_cgto_sym_tensor_fns(cgto, s2, s4)

  if cfg.method_cfg.name == "KS":
    dg = DifferentiableGrids(pyscf_mol)
    dg.level = cfg.intor_cfg.quad_level
  else:
    dg = None

  return cgto_tensor_fns, pyscf_mol, cgto, dg


def incore_cgto_scf(cfg: D4FTConfig) -> None:
  """Solve for ground state of a molecular system with SCF KS-DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore.

  NOTE: since jax-xc doesn't have vxc yet the vxc here is fixed to LDA
  """
  key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)
  cgto_tensor_fns, pyscf_mol, cgto, dg = build_mf_cgto(cfg)
  cgto_e_tensors = cgto_tensor_fns.get_incore_tensors(cgto)
  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  ovlp = get_ovlp(cgto, cgto_e_tensors)

  vxc_fn = get_lda_vxc(
    grids_and_weights, cgto, polarized=not cfg.method_cfg.restricted
  )
  cgto_fock_fn = get_cgto_fock_fn(cgto, cgto_e_tensors, vxc_fn)
  cgto_fock_jit = jax.jit(cgto_fock_fn)

  # get initial mo_coeff
  mo_coeff_fn = partial(cgto.get_mo_coeff, restricted=cfg.method_cfg.restricted)
  mo_coeff_fn = hk.without_apply_rng(hk.transform(mo_coeff_fn))
  params = mo_coeff_fn.init(key)
  mo_coeff = mo_coeff_fn.apply(params, apply_spin_mask=False)

  polarized = not cfg.method_cfg.restricted
  xc_func = get_xc_functional(cfg.method_cfg.xc_type, polarized)
  xc_fn = get_xc_intor(grids_and_weights, cgto, xc_func, polarized)
  kin_fn, ext_fn, har_fn = get_cgto_intor(
    cgto, intor="obsa", cgto_e_tensors=cgto_e_tensors
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

  transpose_axis = (1, 0) if cfg.method_cfg.restricted else (0, 2, 1)

  @jax.jit
  def scf_iter(fock):
    e_orb, mo_coeff = jnp.linalg.eigh(fock)
    mo_coeff = jnp.transpose(mo_coeff, transpose_axis)
    return e_orb, mo_coeff

  fock = jnp.eye(cgto.nao)  # initial guess
  logger = RunLogger()
  for step in range(cfg.solver_cfg.epochs):
    new_fock = cgto_fock_jit(mo_coeff)
    fock = (
      1 - cfg.solver_cfg.momentum
    ) * new_fock + cfg.solver_cfg.momentum * fock
    e_orb, mo_coeff = scf_iter(fock)
    logging.info(f"{e_orb=}")
    residual = jnp.eye(cgto.nao) - mo_coeff[0].T @ ovlp @ mo_coeff[0]
    thresh = np.abs(residual).max()
    energies = energy_fn(mo_coeff * cgto.nocc[:, :, None])
    logger.log_step(energies, step, thresh)
    logger.get_segment_summary()


def cgto_direct_opt(
  cfg: D4FTConfig,
  run_pyscf_benchmark: bool = False,
) -> float:
  """Solve for ground state of a molecular system with direct optimization DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore."""
  key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)

  cgto_tensor_fns, pyscf_mol, cgto, dg = build_mf_cgto(cfg)

  if cfg.intor_cfg.incore:
    cgto_e_tensors = cgto_tensor_fns.get_incore_tensors(cgto)
  else:
    cgto_e_tensors = None

  grids_and_weights = dg.build(pyscf_mol.atom_coords())

  def H_factory() -> Tuple[Callable, Hamiltonian]:
    """Auto-grad scope"""
    ovlp = get_ovlp(cgto, cgto_e_tensors)
    # TODO: out-of-core + basis optimization
    if cfg.solver_cfg.basis_optim != "":
      optimizable_params = cfg.solver_cfg.basis_optim.split(",")
      cgto_hk = cgto.to_hk(optimizable_params)
    else:
      cgto_hk = cgto
    cgto_intor = get_cgto_intor(
      cgto_hk, intor="obsa", cgto_e_tensors=cgto_e_tensors
    )
    mo_coeff_fn = partial(
      cgto_hk.get_mo_coeff,
      restricted=cfg.method_cfg.restricted,
      ortho_fn=qr_factor,
      ovlp_sqrt_inv=sqrt_inv(ovlp),
    )

    if cfg.method_cfg.name == "KS":
      polarized = not cfg.method_cfg.restricted
      xc_func = get_xc_functional(cfg.method_cfg.xc_type, polarized)
      # TODO: fix this to enable geometry optimization
      # NOTE: geometry optimization is not working yet since the function
      # treutler_atomic_radii_adjust is not differentiable yet
      # grids_and_weights = dg.build(cgto_hk.atom_coords)
      xc_fn = get_xc_intor(grids_and_weights, cgto_hk, xc_func, polarized)
      cgto_intor = cgto_intor._replace(xc_fn=xc_fn)

    return mf_cgto(cgto_hk, cgto_intor, mo_coeff_fn)

  # e_total = scipy_opt(cfg.solver_cfg, H_factory, key)
  # breakpoint()

  H_transformed = hk.multi_transform(H_factory)
  params = H_transformed.init(key)
  H = Hamiltonian(*H_transformed.apply)

  logger, traj = sgd(cfg.solver_cfg, H, params, key)

  min_e_step = logger.data_df.e_total.astype(float).idxmin()
  logging.info(f"lowest total energy: \n {logger.data_df.iloc[min_e_step]}")
  lowest_e = logger.data_df.e_total.astype(float).min()

  # NOTE: diagonalize the fock matrix gives a different mo_coeff
  # rdm1 = get_rdm1(traj[-1].mo_coeff)
  # scf_mo_coeff = pyscf_wrapper(
  #   pyscf_mol,
  #   cfg.method_cfg.restricted,
  #   cfg.method_cfg.xc_type,
  #   cfg.intor_cfg.quad_level,
  #   algo="KS",
  #   rdm1=rdm1,
  # )
  # breakpoint()

  if run_pyscf_benchmark:
    pyscf_logger = pyscf_benchmark(
      cfg, pyscf_mol, cgto, cgto_e_tensors, grids_and_weights
    )

    logging.info("energy diff")
    logging.info(
      pyscf_logger.data_df.iloc[-1] - logger.data_df.iloc[min_e_step]
    )
    logging.info("time diff")
    pyscf_e_total = pyscf_logger.data_df.e_total[0]
    e_lower_step = (logger.data_df.e_total < pyscf_e_total).argmax()
    t_total = logger.data_df.iloc[:e_lower_step].time.sum()
    logging.info(f"pyscf time: {pyscf_logger.data_df.time[0]}")
    logging.info(f"d4ft time: {t_total}")

  if cfg.uuid != "":
    logger.save(cfg, "direct_opt")
    with (cfg.get_save_dir() / "traj.pkl").open("wb") as f:
      pickle.dump(traj[-1], f)

  return lowest_e


def incore_cgto_pyscf_benchmark(cfg: D4FTConfig) -> RunLogger:
  cgto_tensor_fns, pyscf_mol, cgto, dg = build_mf_cgto(cfg)
  cgto_e_tensors = cgto_tensor_fns.get_incore_tensors(cgto)
  grids_and_weights = dg.build(pyscf_mol.atom_coords())
  return pyscf_benchmark(
    cfg, pyscf_mol, cgto, cgto_e_tensors, grids_and_weights
  )


def pyscf_benchmark(
  cfg: D4FTConfig,
  pyscf_mol: pyscf.gto.mole.Mole,
  cgto: CGTO,
  cgto_e_tensors,
  grids_and_weights,
) -> RunLogger:
  """Call PySCF to solve for ground state of a molecular system with SCF DFT,
  then load the computed MO coefficients from PySCF and redo the energy integral
  with obsa, where the energy tensors are precomputed/incore."""
  cgto_intor = get_cgto_intor(cgto, intor="obsa", cgto_e_tensors=cgto_e_tensors)
  if cfg.method_cfg.name == "KS":
    polarized = not cfg.method_cfg.restricted
    xc_func = get_xc_functional(cfg.method_cfg.xc_type, polarized)
    xc_fn = get_xc_intor(grids_and_weights, cgto, xc_func, polarized)
    cgto_intor = cgto_intor._replace(xc_fn=xc_fn)

  # solve for ground state with PySCF and get the mo_coeff
  atom_mf, mo_coeff = pyscf_wrapper(
    pyscf_mol,
    cfg.method_cfg.restricted,
    cfg.method_cfg.xc_type,
    cfg.intor_cfg.quad_level,
    method=cfg.method_cfg.name
  )

  # add spin and apply occupation mask
  mo_coeff *= cgto.nocc[:, :, None]

  # eval with d4ft
  _, H = mf_cgto(cgto, cgto_intor, mo_coeff_fn=make_constant_fn(mo_coeff))

  _, (energies, _) = H.energy_fn(mo_coeff)
  e1 = energies.e_kin + energies.e_ext

  logger = RunLogger()  # start timer
  logger.log_step(energies, 0, 0)
  logger.get_segment_summary()
  logging.info(f"1e energy:{e1}")

  # check results
  assert np.allclose(e1, atom_mf.scf_summary['e1'])
  assert np.allclose(energies.e_har, atom_mf.scf_summary['coul'])
  assert np.allclose(energies.e_xc, atom_mf.scf_summary['exc'])

  if cfg.uuid != "":
    logger.save(cfg, "pyscf")
    with (cfg.get_save_dir() / "pyscf_mo_coeff.pkl").open("wb") as f:
      pickle.dump(mo_coeff, f)

  return logger
