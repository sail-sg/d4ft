from absl import logging
import time
import jax
import jax.numpy as jnp
from jdft.energy import _energy_gs, energy_gs
import optax
from jdft.functions import decov
from jdft.ao_int import _ao_ext_int, _ao_kin_int
from jdft.sampler import batch_sampler
import pandas as pd


def sgd(
  mol,
  epoch,
  lr,
  seed=123,
  converge_threshold=1e-3,
  batch_size=1000,
  optimizer='sgd',
  pre_cal=False
):
  epoch_d, e_tot_d, e_kin_d, e_ext_d, e_xc_d, e_hartree_d, \
    e_nuc_d, time_d, acc_time_d = [], [], [], [], [], [], [], [], [0]

  """Run the main training loop."""
  params = mol._init_param(seed)

  if optimizer == 'sgd':
    optimizer = optax.sgd(lr)
  elif optimizer == 'adam':
    optimizer = optax.adam(lr)
  else:
    raise NotImplementedError

  opt_state = optimizer.init(params)

  if batch_size > mol.grids.shape[0]:
    batch_size = mol.grids.shape[0]

  @jax.jit
  def sampler(seed):
    return batch_sampler(mol.grids, mol.weights, batch_size, seed=seed)

  dataset1 = sampler(seed)

  if pre_cal:
    logging.info('Preparing for integration...')
    start = time.time()
    overlap_decov = decov(mol.ao.overlap())

    @jax.jit
    def _ao_kin_mat_fun(batch):
      g, w = batch
      _ao_kin_mat = _ao_kin_int(mol.ao, g, w)
      _ao_kin_mat = overlap_decov @ _ao_kin_mat @ overlap_decov.T

      return _ao_kin_mat

    @jax.jit
    def _ao_ext_mat_fun(batch):
      g, w = batch
      _ao_ext_mat = _ao_ext_int(mol.ao, mol.nuclei, g, w)
      _ao_ext_mat = overlap_decov @ _ao_ext_mat @ overlap_decov.T

      return _ao_ext_mat

    _ao_kin_mat = jnp.zeros([mol.nao, mol.nao])
    _ao_ext_mat = jnp.zeros([mol.nao, mol.nao])

    # for batch in dataset1.as_numpy_iterator():
    for batch in dataset1:

      _ao_kin_mat += _ao_kin_mat_fun(batch)
      _ao_ext_mat += _ao_ext_mat_fun(batch)

    _ao_kin_mat /= len(list(dataset1))
    _ao_ext_mat /= len(list(dataset1))
    logging.info(f"Pre-calculation finished. Time: {(time.time()-start):.3f}")

  @jax.jit
  def update(params, opt_state, batch1, batch2):

    def loss(params):

      def mo(r):
        return mol.mo(params, r) * mol.nocc

      if pre_cal:
        return _energy_gs(
          mo, mol.nuclei, params, _ao_kin_mat, _ao_ext_mat, mol.nocc, batch1,
          batch2
        )
      else:
        return energy_gs(mo, mol.nuclei, batch1, batch2)

    (e_total, e_splits), grad = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, (e_total, *e_splits)

  logging.info(f"Starting... Random Seed: {seed}, Batch size: {batch_size}")

  prev_loss = 0.
  start_time = time.time()
  converged = False

  logging.info(f"Batch size: {batch_size}")
  logging.info(f"Total grid points: {len(mol.grids)}")

  # batch = (mol.grids, mol.weights)
  for i in range(epoch):
    iter_start = time.time()
    Es_batch = []

    batchs1 = sampler(seed + i)
    batchs2 = sampler(seed + i + 1)

    for batch1, batch2 in zip(batchs1, batchs2):
      params, opt_state, Es = update(params, opt_state, batch1, batch2)
      Es_batch.append(Es)
    iter_end = time.time()

    # retrieve all energy terms
    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
      jnp.array(Es_batch), axis=0
    )
    # track total energy for convergence check

    time_d.append(iter_end - iter_start); acc_time_d.append(acc_time_d[-1] + iter_end - iter_start); epoch_d.append(i+1);
    e_tot_d.append(e_total); e_kin_d.append(e_kin); e_ext_d.append(e_ext); e_xc_d.append(e_xc); e_hartree_d.append(e_hartree); e_nuc_d.append(e_nuc);


    if (i + 1) % 1 == 0:
      logging.info(
        f"Iter: {i+1}/{epoch}. \
          Ground State Energy: {e_total:.3f}. \
          Time: {iter_end - iter_start:.3f}"
      )

    if jnp.abs(prev_loss - e_total) < converge_threshold:
      converged = True
      break
    else:
      prev_loss = e_total

  logging.info(
    f"Converged: {converged}. \n"
    f"Total epochs run: {i+1}. \n"
    f"Training Time: {(time.time() - start_time):.3f}s. \n"
  )
  logging.info("Energy:")
  logging.info(f" Ground State: {e_total}")
  logging.info(f" Kinetic: {e_kin}")
  logging.info(f" External: {e_ext}")
  logging.info(f" Exchange-Correlation: {e_xc}")
  logging.info(f" Hartree: {e_hartree}")
  logging.info(f" Nucleus Repulsion: {e_nuc}")

  info_dict = {"epoch":epoch_d, "e_tot":e_tot_d, "e_kin":e_kin_d, "e_ext":e_ext_d, "e_xc":e_xc_d, "e_hartree":e_hartree_d, "e_nuc":e_nuc_d, \
    "time":time_d, "acc_time":acc_time_d[1:]}

  df = pd.DataFrame(data=info_dict, index=None)
  df.to_excel("jdft/results/"+args.geometry+"/"+args.geometry+"_"+str(args.lr)+"_"+str(args.batch_size)+"_"+str(args.seed)+"_gd.xlsx", index=False)

  return params


if __name__ == "__main__":
  import jdft.geometries
  from jdft.molecule import molecule
  import argparse

  logging.set_verbosity(logging.INFO)

  parser = argparse.ArgumentParser(description="JDFT Project")
  parser.add_argument("--batch_size", type=int, default=100000)
  parser.add_argument("--epoch", type=int, default=200)
  parser.add_argument("--converge_threshold", type=float, default=1e-5)
  parser.add_argument("--lr", type=float, default=1e-2)
  parser.add_argument("--pre_cal", type=bool, default=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--geometry", type=str, default="h2o")
  parser.add_argument("--opt", type=str, default="adam")
  parser.add_argument("--basis_set", type=str, default="sto-3g")
  parser.add_argument("--print_coeff", type=bool, default=False)
  args = parser.parse_args()

  geometry = getattr(jdft.geometries, args.geometry+"_geometry")
  mol = molecule(geometry, spin=0, level=1, basis=args.basis_set)
  sgd(
    mol,
    epoch=args.epoch,
    lr=args.lr,
    batch_size=args.batch_size,
    converge_threshold=args.converge_threshold,
    pre_cal=args.pre_cal,
    optimizer=args.opt,
    seed=args.seed
  )
