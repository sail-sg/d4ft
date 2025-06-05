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
import time
from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyscf
from absl import logging

from d4ft.config import D4FTConfig
from d4ft.hamiltonian.mf_cgto import mf_cgto
from d4ft.hamiltonian.ortho import qr_factor
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.gto.utils import Shell
from d4ft.integral.obara_saika.driver import get_cgto_sym_tensor_fns
from d4ft.integral.quadrature.grids import DifferentiableGrids
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.solver.pyscf_wrapper import pyscf_wrapper
from d4ft.solver.scf import scf
from d4ft.solver.sgd import sgd
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.types import Energies, Hamiltonian, TrainingState
from d4ft.xc import get_xc_functional, get_xc_intor


def add_orbital_to_basis(
  mol: Mol,
  angular_momentum: int = 2,  # d-orbital by default
  exponents: Sequence[float] = (2.0, 0.5, 0.1),
  coefficients: Sequence[float] = (1.0, 1.0, 1.0)
) -> Mol:
  """Add an orbital of specified angular momentum to each atom's basis set."""
  # Create new basis with additional orbitals
  new_basis = {}
  for element, basis in mol.basis.items():
    new_basis[element] = list(basis)
    new_basis[element].append(
      [angular_momentum] + [[e, c] for e, c in zip(exponents, coefficients)]
    )

  # Calculate new number of basis functions
  old_nao = mol.nocc.shape[1]  # original number of AOs
  additional_funcs = (2 * angular_momentum +
                      1) * len(mol.elements)  # number of new basis functions
  new_nao = old_nao + additional_funcs

  # Create new occupation array with zeros for new orbitals
  new_nocc = jnp.zeros((2, new_nao))  # (2 spins, new_nao)
  new_nocc = new_nocc.at[:, :old_nao].set(mol.nocc)  # copy old occupations

  # Create new molecule with modified basis and occupations
  return mol._replace(basis=new_basis, nocc=new_nocc)


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

  if cfg.sys_cfg.add_basis != "":
    for new_pgto in cfg.sys_cfg.add_basis.split(","):
      shell, n = new_pgto.split(":")
      l = Shell[shell].value
      n = int(n)
      mol = add_orbital_to_basis(
        mol, angular_momentum=l, exponents=(1.,) * n, coefficients=(1.,) * n
      )

  cgto = CGTO.from_mol(mol)

  # TODO: intor.split() for pmap / batched
  s2 = obsa.angular_static_args(*[cgto.pgto.angular] * 2)
  s4 = obsa.angular_static_args(*[cgto.pgto.angular] * 4)
  cgto_tensor_fns = get_cgto_sym_tensor_fns(cgto, s2, s4)

  dg = DifferentiableGrids(pyscf_mol)

  def H_factory() -> Tuple[Callable, Hamiltonian]:
    """Auto-grad scope"""
    if cfg.solver_cfg.basis_optim != "":
      optimizable_params = cfg.solver_cfg.basis_optim.split(",")
      cgto_hk = cgto.to_hk(optimizable_params)
    else:
      cgto_hk = cgto
    mo_coeff_fn = partial(
      cgto_hk.get_mo_coeff,
      restricted=cfg.method_cfg.restricted,
      ortho_fn=qr_factor,
    )
    xc_fn = None

    if cfg.method_cfg.name == "KS":
      polarized = not cfg.method_cfg.restricted
      xc_func = get_xc_functional(cfg.method_cfg.xc_type, polarized)
      grids_and_weights = dg.build(cgto_hk.atom_coords)
      xc_fn = get_xc_intor(grids_and_weights, cgto_hk, xc_func, polarized)

    return mf_cgto(cgto_hk, cgto_tensor_fns, mo_coeff_fn, xc_fn)

  return pyscf_mol, H_factory, cgto, None


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

  _, H_factory, cgto, cgto_fock_fn = build_mf_cgto(cfg)
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


def init_from_cfg(cfg: D4FTConfig):
  key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)
  _, H_factory, cgto, _ = build_mf_cgto(cfg)
  H_transformed = hk.multi_transform(H_factory)
  params = H_transformed.init(key)
  H = Hamiltonian(*H_transformed.apply)
  opt_states = get_optimizer(cfg.solver_cfg, params, key)
  optimizer, state = opt_states["main"]

  def loss_fn(params, rng_key, cgto_e_tensors) -> float:
    return H.energy_fn(params, rng_key, cgto_e_tensors)

  @partial(jax.jit, static_argnames=("filter_grad",))
  def gd_step(state: TrainingState, cgto_e_tensors, filter_grad=None) -> Tuple:
    """update parameter, and accumulate gradients"""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    val_and_grads_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grad = val_and_grads_fn(state.params, rng_key, cgto_e_tensors)
    energies = aux

    # Filter gradients if filter_grad is provided
    if filter_grad is not None:
      filtered_grad = {k: v for k, v in grad['~'].items() if k in filter_grad}
      # Zero out gradients for parameters not in filter_grad
      grad = {
        '~':
          {
            k: filtered_grad.get(k, jnp.zeros_like(v))
            for k, v in grad['~'].items()
          }
      }

    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(params, opt_state, next_rng_key), energies

  @partial(jax.jit, static_argnames=("debug"))
  def grad_fn(params, rng_key, debug=False):
    params_mo = params["~"]["mo_params"]
    params_center = params["~"]["center"]

    def e_fn(params_center, params_mo, rng_key):
      params = {"~": {"mo_params": params_mo, "center": params_center}}
      if debug:
        return H.nuc_fn(params, rng_key)
      else:
        return H.energy_fn(params, rng_key)

    return jax.grad(e_fn, has_aux=not debug)(params_center, params_mo, rng_key)

  return H, state, gd_step, grad_fn, optimizer


def cgto_direct(
  cfg: D4FTConfig,
  run_pyscf_benchmark: bool = False,
) -> float:
  """Solve for ground state of a molecular system with direct optimization DFT,
  where CGTO basis are used and the energy tensors are precomputed/incore."""
  key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)

  pyscf_mol, H_factory, _, _ = build_mf_cgto(cfg)

  H_transformed = hk.multi_transform(H_factory)
  params = H_transformed.init(key)
  H_hk = Hamiltonian(*H_transformed.apply)

  logger, state = sgd(cfg, H_hk, params, key)

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
    pyscf_benchmark(cfg, pyscf_mol, H_hk, state, logger)

  # if cfg.uuid != "":
  #   logger.save(cfg, "direct_opt")
  #   with (cfg.get_save_dir() / "traj.pkl").open("wb") as f:
  #     pickle.dump(traj[-1], f)

  return lowest_e


def incore_cgto_pyscf_benchmark(cfg: D4FTConfig) -> RunLogger:
  assert cfg.intor_cfg.incore
  pyscf_mol, H_factory, _, _ = build_mf_cgto(cfg)
  return pyscf_benchmark(cfg, pyscf_mol, H_factory())


def pyscf_benchmark(
  cfg: D4FTConfig,
  pyscf_mol: pyscf.gto.mole.Mole,
  H: Hamiltonian,
  state: Optional[TrainingState] = None,
  compare_logger: Optional[RunLogger] = None,
) -> RunLogger:
  """Call PySCF to solve for ground state of a molecular system with SCF DFT,
  then load the computed MO coefficients from PySCF and redo the energy integral
  with obsa, where the energy tensors are precomputed/incore."""
  # solve for ground state with PySCF and get the mo_coeff
  start_time = time.time()
  atom_mf, pyscf_mo_coeff = pyscf_wrapper(pyscf_mol, cfg)
  end_time = time.time()
  pyscf_time = end_time - start_time

  # add spin and apply occupation mask
  nocc = Mol.get_nocc(pyscf_mol)
  pyscf_mo_coeff *= nocc[:, :, None]

  e_fn_jit = jax.jit(H.energy_fn)
  _, pyscf_energies = e_fn_jit(
    state.params, state.rng_key, mo_coeff=pyscf_mo_coeff
  )

  e1 = pyscf_energies.e_kin + pyscf_energies.e_ext

  logger = RunLogger()
  logger.log_step(pyscf_energies, 0, 0)
  logger.get_segment_summary()

  # check integration results
  assert np.allclose(e1, atom_mf.scf_summary['e1'])
  if cfg.method_cfg.name == "KS":
    assert np.allclose(pyscf_energies.e_har, atom_mf.scf_summary['coul'])
    assert np.allclose(pyscf_energies.e_xc, atom_mf.scf_summary['exc'])
  elif cfg.method_cfg.name == "HF":
    e2_hf = pyscf_energies.e_har + pyscf_energies.e_xc
    assert np.allclose(e2_hf, atom_mf.scf_summary['e2'])

  if cfg.uuid != "":
    logger.save(cfg, "pyscf")
    with (cfg.get_save_dir() / "pyscf_mo_coeff.pkl").open("wb") as f:
      pickle.dump(pyscf_mo_coeff, f)

  if compare_logger is not None:
    logging.info("energy diff")
    min_e_step = compare_logger.data_df.e_total.astype(float).idxmin()
    logging.info(
      logger.data_df.iloc[-1] - compare_logger.data_df.iloc[min_e_step]
    )
    logging.info("time diff")
    t_total = compare_logger.data_df.time.sum()
    logging.info(f"pyscf time: {pyscf_time}")
    logging.info(f"d4ft time: {t_total}")

  return logger
