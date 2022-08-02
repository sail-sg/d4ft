import jdft
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

  @jax.jit
  def update(params, opt_state, grids, weights):

    def loss(params):

      def mo(r):
        return mol.mo(params, r) * mol.nocc

      return energy_gs(mo, mol.nuclei, grids, weights)

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
  dataset = tf.data.Dataset.from_tensor_slices((
    mol.grids,
    mol.weights,
  )).shuffle(len(mol.grids)).batch(batch_size)

  for i in range(epoch):
    Es_batch = []
    for g, w in dataset.as_numpy_iterator():
      w = w * len(mol.grids) / w.shape[0]
      params, opt_state, Es = update(params, opt_state, g, w)
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
    f"Converged: {converged}."
    f"Total epochs run: {i+1}."
    f"Training Time: {(time.time() - start_time):.3f}s."
  )
  logging.info("Energy:")
  logging.info(f" Ground State: {e_total}")
  logging.info(f" Kinetic: {e_kin}")
  logging.info(f" External: {e_ext}")
  logging.info(f" Exchange-Correlation: {e_xc}")
  logging.info(f" Hartree: {e_hartree}")
  logging.info(f" Nucleus Repulsion: {e_nuc}")


if __name__ == "__main__":
  from jdft.geometries import h2o_geometry

  mol = jdft.molecule(h2o_geometry, spin=0, level=1, basis="6-31g")
  train(mol, epoch=100, lr=1e-2, batch_size=2000, converge_threshold=1e-5)
