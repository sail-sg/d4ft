"""Solve DFT with gradient descent"""

from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import pyscf
from absl import logging

from d4ft.config import DFTConfig, OptimizerConfig
from d4ft.hamiltonian import Hamiltonian, calc_hamiltonian
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.types import Array, Energies, Grads, Trajectory, Transition
from d4ft.utils import compose


def sgd_solver(
  dft_cfg: DFTConfig, optim_cfg: OptimizerConfig, mol: pyscf.gto.mole.Mole,
  key: Array
) -> Tuple[float, Trajectory, Hamiltonian]:
  # init Stiefel manifold params
  nmo = mol.nao
  shape = ([nmo, nmo] if dft_cfg.rks else [2, nmo, nmo])
  params = jax.random.normal(key, shape) / jnp.sqrt(nmo)

  H = calc_hamiltonian(mol, dft_cfg)

  # init optimizer
  optimizer = get_optimizer(optim_cfg)
  opt_state = optimizer.init(params)

  e_fns = [H.kin_fn, H.ext_fn, H.eri_fn, H.xc_fn]

  @jax.jit
  def update(params, opt_state):
    """update parameter, and accumulate gradients"""
    val_and_grads = [
      jax.value_and_grad(compose(e_fn, H.mo_coeff_fn))(params) for e_fn in e_fns
    ]  # (nao, nao)

    # get gradient w.r.t. mo_coeff
    mo_coeff = H.mo_coeff_fn(params)  # (2*nao, nao)
    e_grads = [jax.grad(e_fn)(mo_coeff) for e_fn in e_fns]  # (2*nao, nao)

    # to get total gradient, compose with mo_grad
    # mo_grad = jax.jacfwd(H.mo_coeff_fn)(params)  # (2*nao, nao, nao, nao)
    # t_grad = jnp.einsum("ijkl,ij->kl", mo_grad, e_grads[0])
    # np.allclose(t_grad, val_and_grads[0][1], atol=1e-5)

    vals, grads = zip(*val_and_grads)
    e_kin, e_ext, e_eri, e_xc = vals
    grad = sum(grads)
    e_total = e_kin + e_ext + e_eri + e_xc + H.e_nuc
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    energies = Energies(e_total, e_kin, e_ext, e_eri, e_xc, H.e_nuc)
    grads = Grads(*e_grads)
    return params, opt_state, energies, grads

  # GD loop
  traj = []
  prev_e_total = 0.
  converged = False
  logger = RunLogger()
  for step in range(optim_cfg.epochs):
    new_params, opt_state, energies, grads = update(params, opt_state)

    logger.log_step(energies, step)
    segment_df = logger.get_segment_summary()
    e_total = segment_df.e_total.mean()

    mo_coeff = H.mo_coeff_fn(params)
    t = Transition(mo_coeff, energies, grads)
    traj.append(t)

    params = new_params

    # check convergence
    e_total = energies.e_total
    if jnp.abs(prev_e_total - e_total) < dft_cfg.converge_threshold:
      converged = True
      break
    else:
      prev_e_total = e_total

  logging.info(f"Converged: {converged}")

  return e_total, traj, H
