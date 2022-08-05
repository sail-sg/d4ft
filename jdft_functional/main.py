from absl import logging
import time
import jax
import jax.numpy as jnp
import tensorflow as tf
from energy import _energy_gs, energy_gs
import optax
from jdft.functions import decov
from ao_int import _ao_ext_int, _ao_kin_int


def train(
    mol,
    epoch,
    lr,
    seed=123,
    converge_threshold=1e-3,
    batch_size=1000,
    pre_cal=False
):
  """Run the main training loop."""
  params = mol._init_param(seed)
  optimizer = optax.sgd(lr)
  opt_state = optimizer.init(params)

  if batch_size > mol.grids.shape[0]:
    batch_size = mol.grids.shape[0]

  def reweigt(batch):
    g, w = batch
    w = w * len(mol.grids) / w.shape[0]
    return g, w

  dataset1 = tf.data.Dataset.from_tensor_slices((
      mol.grids,
      mol.weights,
  )).shuffle(
      len(mol.grids), seed=seed
  ).batch(
      batch_size, drop_remainder=True
  )

  dataset2 = tf.data.Dataset.from_tensor_slices((
      mol.grids,
      mol.weights,
  )).shuffle(
      len(mol.grids), seed=seed + 1
  ).batch(
      batch_size, drop_remainder=True
  )

  if pre_cal:
    logging.info('Preparing for integration...')
    start = time.time()
    overlap_decov = decov(mol.ao.overlap())

    @jax.jit
    def _ao_kin_mat_fun(batch):
      g, w = reweigt(batch)
      _ao_kin_mat = _ao_kin_int(mol.ao, g, w)
      _ao_kin_mat = overlap_decov @ _ao_kin_mat @ overlap_decov.T

      return _ao_kin_mat

    @jax.jit
    def _ao_ext_mat_fun(batch):
      g, w = reweigt(batch)
      _ao_ext_mat = _ao_ext_int(mol.ao, mol.nuclei, g, w)
      _ao_ext_mat = overlap_decov @ _ao_ext_mat @ overlap_decov.T

      return _ao_ext_mat

    _ao_kin_mat = jnp.zeros([mol.nao, mol.nao])
    _ao_ext_mat = jnp.zeros([mol.nao, mol.nao])

    for batch in dataset1.as_numpy_iterator():

      _ao_kin_mat += _ao_kin_mat_fun(batch)
      _ao_ext_mat += _ao_ext_mat_fun(batch)

    _ao_kin_mat /= len(list(dataset1.as_numpy_iterator()))
    _ao_ext_mat /= len(list(dataset1.as_numpy_iterator()))
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
  e_train = []
  converged = False

  logging.info(f"Batch size: {batch_size}")
  logging.info(f"Total grid points: {len(mol.grids)}")

  for i in range(epoch):
    Es_batch = []

    for batch1, batch2 in zip(
        dataset1.as_numpy_iterator(), dataset2.as_numpy_iterator()
    ):
      batch1 = reweigt(batch1)
      batch2 = reweigt(batch2)
      params, opt_state, Es = update(params, opt_state, batch1, batch2)
      Es_batch.append(Es)

    # retrieve all energy terms
    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
        jnp.array(Es_batch), axis=0
    )
    # track total energy for convergence check
    e_train.append(e_total)

    if (i + 1) % 1 == 0:
      logging.info(f"Iter: {i+1}/{epoch}. Ground State Energy: {e_total:.3f}.")

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

  return params


if __name__ == "__main__":
  from jdft.geometries import c60_geometry
  from molecule import molecule

  mol = molecule(c60_geometry, spin=0, level=1, basis="6-31g")
  train(
      mol,
      epoch=100,
      lr=1e-2,
      batch_size=5000,
      converge_threshold=1e-5,
      pre_cal=True
  )
