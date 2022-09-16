import jax
import jax.numpy as jnp
from energy import wave2density, integrand_kinetic, \
  integrand_external, integrand_xc_lda, e_nuclear
from typing import Callable
from absl import logging
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from jdft.functions import decov
from jdft.ao_int import _ao_ext_int, _ao_kin_int
import os
from jdft.sampler import batch_sampler
from jdft.energy import _energy_gs, energy_gs

logging.set_verbosity(logging.INFO)


def hamil_kinetic(ao: Callable, batch):
  r"""
  \int \phi_i \nabla^2 \phi_j dx
  Args:
    mo (Callable):
    batch: a tuple of (grids, weights)
  Return:
    [2, N, N] Array.
  """
  return integrate_s(integrand_kinetic(ao, True), batch)


def energy_kinetic(mo: Callable, batch):
  return integrate_s(integrand_kinetic(mo), batch)


def hamil_external(ao: Callable, nuclei, batch):
  r"""
  \int \phi_i \nabla^2 \phi_j dx
  Args:
    mo (Callable): a [3] -> [2, N] function
    batch: a tuple (grids, weights)
  Return:
    [2, N, N] Array
  """
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']

  def v(r):
    return -jnp.sum(
      nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + 1e-20)
    )

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: v(r) * m(r), batch)


def energy_external(mo: Callable, nuclei, batch):
  return integrate_s(integrand_external(mo, nuclei), batch)


def hamil_hartree(ao: Callable, mo_old, batch1, batch2):
  density = wave2density(mo_old)

  def g(r):
    r"""
      g(r) = \int n(r')/|r-r'| dr'
    """

    def v(x):
      return density(x) / jnp.sqrt(
        jnp.sum((x - r)**2) + 1e-10
      ) * jnp.where(jnp.all(x == r), 0, 1)

    return integrate_s(v, batch1)

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: g(r) * m(r), batch2)


def energy_hartree(mo_old: Callable, batch):
  r"""
  Return n(x)n(y)/|x-y|
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
    mo only takes one argment, which is the coordinate.
  Return:
    a function: [3] x [3] -> [1]
  """
  density = wave2density(mo_old)

  def g(r):
    r"""
      g(r) = \int n(r')/|r-r'| dr'
    """

    def v(x):
      return density(x) / jnp.sqrt(
        jnp.sum((x - r)**2) + 1e-10
      ) * jnp.where(jnp.all(x == r), 0, 1)
      """return density(x) / jnp.clip(
        jnp.linalg.norm(x - r), a_min=1e-8
      ) * jnp.where(jnp.all(x == r), 2e-9, 1)"""

    return integrate_s(v, batch)

  return integrate_s(lambda r: 0.5 * g(r) * density(r), batch)


def hamil_lda(ao: Callable, mo_old, batch):
  """
  v_xc = -(3/pi n(r))^(1/3)
  Return:
    [2, N, N] array
  """
  density = wave2density(mo_old)

  def g(n):
    return -(3 / jnp.pi * n)**(1 / 3)

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: g(density(r)) * m(r), batch)


def energy_lda(mo_old, batch):
  return integrate_s(integrand_xc_lda(mo_old), batch)


def integrate_s(integrand: Callable, batch):
  '''
  Integrate a [3] -> [2, N, ...] function.
  '''
  g, w = batch

  def v(r, w):
    return integrand(r) * w

  @jax.jit
  def f(g, w):
    return jnp.sum(jax.vmap(v)(g, w), axis=0)

  @jax.jit
  def minibatch_f(g, w):
    return jnp.sum(minibatch_vmap(v, batch_size=args.batch_size)(g, w), axis=0)

  return f(g, w) if args.mini_batch is False else minibatch_f(g, w)


def minibatch_vmap(f, in_axes=0, batch_size=10):
  batch_f = jax.vmap(f, in_axes=in_axes)

  def _minibatch_vmap_f(*args):
    nonlocal in_axes
    if not isinstance(in_axes, (tuple, list)):
      in_axes = (in_axes,) * len(args)
    for i, ax in enumerate(in_axes):
      if ax is not None:
        num = args[i].shape[ax]
    num_shards = int(np.ceil(num / batch_size))
    size = num_shards * batch_size
    indices = jnp.arange(0, size, batch_size)

    def _process_batch(start_index):
      batch_args = (
        jax.lax.dynamic_slice_in_dim(
          a,
          start_index=start_index,
          slice_size=batch_size,
          axis=ax,
        ) if ax is not None else a for a, ax in zip(args, in_axes)
      )
      return batch_f(*batch_args)

    def _sum_process_batch(start_index):
      return jnp.sum(_process_batch(start_index), axis=0)

    out = jax.lax.map(_sum_process_batch, indices)
    if isinstance(out, jnp.ndarray):
      out = jnp.reshape(out, (-1, *out.shape[1:]))[:num]
    elif isinstance(out, (tuple, list)):
      out = tuple(jnp.reshape(o, (-1, *o.shape[1:]))[:num] for o in out)
    return out

  return _minibatch_vmap_f


def sscf(mol):
  epoch_d, e_tot_d, e_kin_d, e_ext_d, e_xc_d, e_hartree_d, \
    e_nuc_d, time_d, acc_time_d = [], [], [], [], [], [], [], [], [0]

  batch = (mol.grids, mol.weights)
  params = mol._init_param(args.seed)
  mo_params, _ = params
  _diag_one_ = jnp.ones([2, mol.mo.nmo])
  _diag_one_ = jax.vmap(jnp.diag)(_diag_one_)

  def ao(r):
    return mol.mo((_diag_one_, None), r)

  @jax.jit
  def get_fork(mo_params, batch1, batch2):
    def mo_old(r):
      return mol.mo((mo_params, None), r) * mol.nocc
    return hamil_hartree(ao, mo_old, batch1, batch1) + hamil_lda(ao, mo_old, batch1)


  logging.info('Preparing for integration...')
  start = time.time()

  @jax.jit
  def _hamil_kinetic(batch):
    r"""
    \int \phi_i \nabla^2 \phi_j dx
    Args:
      mo (Callable):
      batch: a tuple of (grids, weights)
    Return:
      [2, N, N] Array.
    """
    return integrate_s(integrand_kinetic(ao, True), batch)

  
  @jax.jit
  def _hamil_external(batch):
    r"""
    \int \phi_i \nabla^2 \phi_j dx
    Args:
      mo (Callable): a [3] -> [2, N] function
      batch: a tuple (grids, weights)
    Return:
      [2, N, N] Array
    """
    nuclei_loc = mol.nuclei['loc']
    nuclei_charge = mol.nuclei['charge']

    def v(r):
      return -jnp.sum(
        nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + 1e-16)
      )

    def m(r):
      return jax.vmap(jnp.outer)(ao(r), ao(r))

    return integrate_s(lambda r: v(r) * m(r), batch)

  _h_kin = _hamil_kinetic(batch)
  _h_ext = _hamil_external(batch)

  fork = _h_kin + _h_ext

  print(f"Pre-calculation finished. Time: {(time.time()-start):.3f}")

  @jax.jit
  def sampler(seed):
    return batch_sampler(mol.grids, mol.weights, args.batch_size, seed=seed)


  def update(mo_params, fock_old):
    batchs1 = sampler(args.seed)
    batchs2 = sampler(args.seed + 1)
    H_batch = []
    for batch1, batch2 in zip(batchs1, batchs2):
      H_batch.append(get_fork(mo_params, batch1, batch2))
    fork = jnp.mean(jnp.array(H_batch), axis=0)
    fork += (_h_ext + _h_kin)

    fork = (1 - args.fock_momentum) * fork + args.fock_momentum * fock_old

    _, mo_params = jnp.linalg.eigh(fork)
    mo_params = jnp.transpose(mo_params, (0, 2, 1))

    return mo_params, fork


  @jax.jit
  def get_energy(mo_params, batch):
    def mo_old(r):
      return mol.mo((mo_params, None), r) * mol.nocc
    e_kin = energy_kinetic(mo_old, batch)
    e_ext = energy_external(mo_old, mol.nuclei, batch)
    e_hartree = energy_hartree(mo_old, batch)
    e_xc = energy_lda(mo_old, batch)
    e_nuc = e_nuclear(mol.nuclei)
    e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc
    return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


  @jax.jit
  def _energy_gs(mo_params, batch1, batch2):
    def mo_old(r):
      return mol.mo((mo_params, None), r) * mol.nocc
    return energy_gs(mo_old, mol.nuclei, batch1, batch2)


  def _get_energy(mo_params):
    batchs1 = sampler(args.seed)
    batchs2 = sampler(args.seed + 1)
    Es_batch = []

    for batch1, batch2 in zip(batchs1, batchs2):
      e_total, e_splits = _energy_gs(mo_params, batch1, batch2)
      Es = (e_total, *e_splits)
      Es_batch.append(Es)

    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
      jnp.array(Es_batch), axis=0
    )

    return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


  # the main loop.
  logging.info(f'{" Starting...SCF loop"}')
  for i in range(args.epoch):
    #if i == 0:
    #  args.fock_momentum = 0

    iter_start = time.time()
    mo_params, fork = update(mo_params, fork)
    iter_end = time.time()

    #args.fock_momentum = 0.9


    e_start = time.time()
    if args.fast_e:
      Es = _get_energy(mo_params)
    else:
      Es = get_energy(mo_params, batch)
    e_total, e_splits = Es
    e_kin, e_ext, e_xc, e_hartree, e_nuc = e_splits
    e_end = time.time()

    logging.info(f" Iter: {i+1}/{args.epoch}")
    logging.info(f" Ground State: {e_total}")
    logging.info(f" Kinetic: {e_kin}")
    logging.info(f" External: {e_ext}")
    logging.info(f" Exchange-Correlation: {e_xc}")
    logging.info(f" Hartree: {e_hartree}")
    logging.info(f" Nucleus Repulsion: {e_nuc}")
    logging.info(f" One Iteration Time: {iter_end - iter_start}")
    logging.info(f" Energy Time: {e_end - e_start}")

    time_d.append(iter_end - iter_start)
    acc_time_d.append(acc_time_d[-1] + iter_end - iter_start)
    epoch_d.append(i + 1)
    e_tot_d.append(e_total)
    e_kin_d.append(e_kin)
    e_ext_d.append(e_ext)
    e_xc_d.append(e_xc)
    e_hartree_d.append(e_hartree)
    e_nuc_d.append(e_nuc)

  info_dict = {
    "epoch": epoch_d,
    "e_tot": e_tot_d,
    "e_kin": e_kin_d,
    "e_ext": e_ext_d,
    "e_xc": e_xc_d,
    "e_hartree": e_hartree_d,
    "e_nuc": e_nuc_d,
    "time": time_d,
    "acc_time": acc_time_d[1:]
  }

  df = pd.DataFrame(data=info_dict, index=None)
  if args.pre_cal:
    df.to_excel(
      "jdft/results/" + args.geometry + "/" + args.geometry + "_" +
      str(args.batch_size) + "_" + str(args.seed) + "_precal_sscf.xlsx",
      index=False
    )
  else:
    df.to_excel(
      "jdft/results/" + args.geometry + "/" + args.geometry + "_" +
      str(args.batch_size) + "_" + str(args.seed) + "_sscf.xlsx",
      index=False
    )


if __name__ == '__main__':
  import jdft.geometries
  from molecule import molecule
  import argparse

  parser = argparse.ArgumentParser(description="JDFT Project")
  parser.add_argument("--batch_size", type=int, default=5000)
  parser.add_argument("--mini_batch", type=bool, default=True)
  parser.add_argument("--epoch", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--geometry", type=str, default="benzene")
  parser.add_argument("--basis_set", type=str, default="sto-3g")
  parser.add_argument("--fock_momentum", type=float, default=0.9)
  parser.add_argument("--pre_cal", type=bool, default=True)
  parser.add_argument("--fast_e", type=bool, default=True)
  parser.add_argument("--batch_size_e", type=int, default=0)
  parser.add_argument("--device", type=str, default="0")
  args = parser.parse_args()

  geometry = getattr(jdft.geometries, args.geometry + "_geometry")
  os.environ["CUDA_VISIBLE_DEVICES"] = args.device

  mol = molecule(geometry, spin=0, level=1, mode="scf", basis=args.basis_set)

  sscf(mol)
