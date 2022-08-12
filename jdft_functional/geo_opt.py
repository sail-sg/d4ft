from absl import logging
import time
import jax
import jax.numpy as jnp
import tensorflow as tf
from energy import energy_gs
import optax

from jdft.functions import distmat


def geo_opt(mol,
            epoch,
            lr,
            seed=123,
            converge_threshold=1e-3,
            batch_size=1000,
            ):
  """Run the main training loop."""
  params = mol._init_param(seed)
  print(params)

  optimizer = optax.adam(lr)
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

  nuclei = mol.nuclei

  params_mask = jnp.ones_like(params[1])
  params_mask = params_mask.at[:, 0].set(0)
  params_mask = params_mask.at[:, 1].set(0)

  @jax.jit
  def update(params, opt_state, batch1, batch2):

    def loss(params):
      mo_params, ao_params = params
      ao_params *= params_mask
      params = (mo_params, ao_params)

      def mo(r):
        return mol.mo(params, r, g=mol.grids, w=mol.weights) * mol.nocc

      nuclei['loc'] = ao_params
      return energy_gs(mo, nuclei, batch1, batch2)

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
      _, coords = params
      logging.info(f"Iter: {i+1}/{epoch}. Ground State Energy: {e_total:.3f}.")
      logging.info(f"current coords: {coords.tolist()}")
      logging.info(f'{distmat(coords)*0.529177249}')

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
  from jdft.geometries import h2_geometry
  from molecule import molecule
  print(h2_geometry)

  h2_start = """
  H 0.0000 0.0000 0.0000;
  H 0.0000 0.0000 1.2;
  """

  mol = molecule(h2_start, spin=0, level=3, basis="6-31g", mode='go')
  geo_opt(
      mol,
      epoch=1000,
      lr=1e-3,
      batch_size=1000,
      converge_threshold=1e-7,
  )
