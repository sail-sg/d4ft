from absl import logging
import time
import jax
import jax.numpy as jnp
import tensorflow as tf
from energy import energy_gs
import optax


def train(mol, epoch, lr, seed=123, converge_threshold=1e-3, batch_size=1000):
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

  @jax.jit
  def update(params, opt_state, batch1, batch2):
    g, w = batch1

    def loss(params):

      def mo(r):
        return mol.mo(params, r, grids=g, weights=w) * mol.nocc

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
  from jdft.geometries import co2_geometry
  from molecule import molecule

  mol = molecule(
      co2_geometry, spin=0, level=1, basis="gaussian", mode='vpf', layers='ococ'
  )
  params = train(
      mol, epoch=200, lr=1e-3, batch_size=20000, converge_threshold=1e-5
  )
  print(params[2])