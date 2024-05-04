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
import numpy as np
import pyscf
from absl import logging

from d4ft.config import D4FTConfig
from d4ft.hamiltonian.cgto_intors import get_cgto_fock_fn, get_cgto_intor
from d4ft.hamiltonian.mf_cgto import mf_cgto
from d4ft.hamiltonian.nuclear import e_nuclear
from d4ft.hamiltonian.ortho import qr_factor, sqrt_inv
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import get_cgto_sym_tensor_fns
from d4ft.integral.quadrature.grids import DifferentiableGrids
from d4ft.logger import RunLogger
from d4ft.solver.pyscf_wrapper import pyscf_wrapper
from d4ft.solver.scf import scf
from d4ft.solver.sgd import sgd
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.types import Hamiltonian
from d4ft.xc import get_lda_vxc, get_xc_functional, get_xc_intor


def build_mf_cgto(cfg: D4FTConfig):
  """Build the CGTO basis with intor for the mean-field calculations
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
    grids_and_weights = dg.build(pyscf_mol.atom_coords())
  else:
    grids_and_weights = None

  if cfg.intor_cfg.incore:
    cgto_e_tensors = cgto_tensor_fns.get_incore_tensors(cgto)

  vxc_ab_fn = get_lda_vxc(
    grids_and_weights, cgto, polarized=not cfg.method_cfg.restricted
  )
  if cfg.intor_cfg.incore:
    cgto_fock_fn = get_cgto_fock_fn(cgto, cgto_e_tensors, vxc_ab_fn)
  else:
    cgto_fock_fn = None

  # # DEBUG
  # g1 = jax.grad(partial(e_nuclear, charge=cgto.charge))(cgto.atom_coords)
  # g2 = jax.jacfwd(partial(e_nuclear, charge=cgto.charge))(cgto.atom_coords)
  # print(g1, g2)
  # breakpoint()

  def H_factory(with_mo_coeff: bool = True) -> Tuple[Callable, Hamiltonian]:
    """Auto-grad scope"""
    if cfg.solver_cfg.basis_optim != "":
      optimizable_params = cfg.solver_cfg.basis_optim.split(",")
      cgto_hk = cgto.to_hk(optimizable_params)
    else:
      cgto_hk = cgto
    if cfg.intor_cfg.incore:
      cgto_intor = get_cgto_intor(
        cgto_hk,
        cgto_e_tensors=cgto_e_tensors,
        intor=cfg.intor_cfg.intor,
      )
    else:
      cgto_intor = get_cgto_intor(
        cgto_hk,
        cgto_tensor_fns=cgto_tensor_fns,
        intor=cfg.intor_cfg.intor,
      )
    if with_mo_coeff:
      mo_coeff_fn = partial(
        cgto_hk.get_mo_coeff,
        restricted=cfg.method_cfg.restricted,
        ortho_fn=qr_factor,
        ovlp_sqrt_inv=sqrt_inv(cgto_intor.ovlp_fn()),
      )
    else:
      mo_coeff_fn = None
    vxc_fn = None
    if cfg.method_cfg.name == "KS":
      polarized = not cfg.method_cfg.restricted
      xc_func = get_xc_functional(cfg.method_cfg.xc_type, polarized)
      # TODO: fix this to enable geometry optimization
      # NOTE: geometry optimization is not working yet since the function
      # treutler_atomic_radii_adjust is not differentiable yet
      # grids_and_weights = dg.build(cgto_hk.atom_coords)
      xc_fn = get_xc_intor(grids_and_weights, cgto_hk, xc_func, polarized)
      cgto_intor = cgto_intor._replace(xc_fn=xc_fn)

      # TODO: figure out the correct loss for vxc
      # vxc_ab_fn = get_lda_vxc(
      #   grids_and_weights, cgto, polarized=not cfg.method_cfg.restricted
      # )
      # vxc_fn = get_vxc_intor(vxc_ab_fn)

    return mf_cgto(cgto_hk, cgto_intor, mo_coeff_fn, vxc_fn=vxc_fn)

  return pyscf_mol, H_factory, cgto, cgto_fock_fn


def incore_cgto_scf(
  cfg: D4FTConfig,
  run_pyscf_benchmark: bool = False,
) -> None:
  """Solve for ground state of a molecular system with SCF KS-DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore.

  NOTE: since jax-xc doesn't have vxc yet the vxc here is fixed to LDA
  """
  assert cfg.intor_cfg.incore
  key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)

  pyscf_mol, H_factory, cgto, cgto_fock_fn = build_mf_cgto(cfg)
  H = H_factory(with_mo_coeff=False)[1]
  ovlp = H.cgto_intors.ovlp_fn()
  cgto_fock_jit = jax.jit(cgto_fock_fn)

  mo_coeff = cgto.get_mo_coeff(
    cfg.method_cfg.restricted, use_hk=False, key=key, apply_spin_mask=False
  )

  energy_fn = H.energy_fn
  energy_fn_jit = jax.jit(energy_fn)

  scf(
    cfg.solver_cfg, cgto, mo_coeff, ovlp, cgto_fock_jit, energy_fn_jit,
    cfg.method_cfg.restricted
  )


def cgto_direct(
  cfg: D4FTConfig,
  run_pyscf_benchmark: bool = False,
) -> float:
  """Solve for ground state of a molecular system with direct optimization DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore."""
  key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)

  pyscf_mol, H_factory, cgto, _ = build_mf_cgto(cfg)

  H_transformed = hk.multi_transform(H_factory)
  params = H_transformed.init(key)
  H_hk = Hamiltonian(*H_transformed.apply)

  logger, traj = sgd(cfg.solver_cfg, H_hk, cgto, params, key)

  min_e_step = logger.data_df.e_total.astype(float).idxmin()
  logging.info(f"lowest total energy: \n {logger.data_df.iloc[min_e_step]}")
  lowest_e = logger.data_df.e_total.astype(float).min()

  # # NOTE: diagonalize the fock matrix gives a different mo_coeff
  # from d4ft.utils import get_rdm1
  # rdm1 = get_rdm1(traj[-1].mo_coeff)
  # atom_mf, scf_mo_coeff = pyscf_wrapper(
  #   pyscf_mol,
  #   cfg.method_cfg.restricted,
  #   cfg.method_cfg.xc_type,
  #   cfg.intor_cfg.quad_level,
  #   method="KS",
  #   rdm1=rdm1,
  # )
  # breakpoint()

  if run_pyscf_benchmark:
    assert cfg.intor_cfg.incore
    pyscf_benchmark(
      cfg, pyscf_mol, H_factory(with_mo_coeff=False)[1], compare_logger=logger
    )

  if cfg.uuid != "":
    logger.save(cfg, "direct_opt")
    with (cfg.get_save_dir() / "traj.pkl").open("wb") as f:
      pickle.dump(traj[-1], f)

  return lowest_e


def incore_cgto_pyscf_benchmark(cfg: D4FTConfig) -> RunLogger:
  assert cfg.intor_cfg.incore
  pyscf_mol, H_factory, _, _ = build_mf_cgto(cfg)
  return pyscf_benchmark(cfg, pyscf_mol, H_factory(with_mo_coeff=False)[1])


def pyscf_benchmark(
  cfg: D4FTConfig,
  pyscf_mol: pyscf.gto.mole.Mole,
  H,
  compare_logger: Optional[RunLogger] = None,
) -> RunLogger:
  """Call PySCF to solve for ground state of a molecular system with SCF DFT,
  then load the computed MO coefficients from PySCF and redo the energy integral
  with obsa, where the energy tensors are precomputed/incore."""
  # solve for ground state with PySCF and get the mo_coeff
  atom_mf, mo_coeff = pyscf_wrapper(pyscf_mol, cfg)

  # add spin and apply occupation mask
  nocc = Mol.get_nocc(pyscf_mol)
  mo_coeff *= nocc[:, :, None]

  _, (energies, _) = H.energy_fn(mo_coeff)
  e1 = energies.e_kin + energies.e_ext

  logger = RunLogger()  # start timer
  logger.log_step(energies, 0, 0)
  logger.get_segment_summary()
  logging.info(f"1e energy:{e1}")

  # check results
  assert np.allclose(e1, atom_mf.scf_summary['e1'])

  if cfg.method_cfg.name == "KS":
    assert np.allclose(energies.e_har, atom_mf.scf_summary['coul'])
    assert np.allclose(energies.e_xc, atom_mf.scf_summary['exc'])
  elif cfg.method_cfg.name == "HF":
    e2_hf = energies.e_har + energies.e_xc
    assert np.allclose(e2_hf, atom_mf.scf_summary['e2'])

  if cfg.uuid != "":
    logger.save(cfg, "pyscf")
    with (cfg.get_save_dir() / "pyscf_mo_coeff.pkl").open("wb") as f:
      pickle.dump(mo_coeff, f)

  if compare_logger is not None:
    logging.info("energy diff")
    min_e_step = compare_logger.data_df.e_total.astype(float).idxmin()
    logging.info(
      logger.data_df.iloc[-1] - compare_logger.data_df.iloc[min_e_step]
    )
    logging.info("time diff")
    pyscf_e_total = logger.data_df.e_total[0]
    e_lower_step = (compare_logger.data_df.e_total < pyscf_e_total).argmax()
    t_total = compare_logger.data_df.iloc[:e_lower_step].time.sum()
    logging.info(f"pyscf time: {logger.data_df.time[0]}")
    logging.info(f"d4ft time: {t_total}")

  return logger
