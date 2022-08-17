from absl import logging
import time
import jax
import jax.numpy as jnp
from energy import energy_gs
import optax

from jdft.functions import distmat


def geo_opt(
  mol,
  epoch,
  lr,
  seed=123,
  converge_threshold=1e-3,
  batch_size=1000,
):
  """Run the main training loop."""
  params = mol._init_param(seed)
  print(params)

  nuclei = mol.nuclei
  optimizer = optax.adam(lr)
  opt_state = optimizer.init((params[0], nuclei['loc']))

  if batch_size > mol.grids.shape[0]:
    batch_size = mol.grids.shape[0]

  batch = (mol.grids, mol.weights)
  params_mask = jnp.ones_like(nuclei['loc'])
  params_mask = params_mask.at[:, 0].set(0)
  params_mask = params_mask.at[:, 1].set(0)

  @jax.jit
  def update(params, nuclei, opt_state, batch):  # params only contains mo.

    mo_params, _ = params
    _params = (mo_params, nuclei['loc'])
    g, w = batch

    def loss(_params):
      mo_params, atom_coords = _params
      atom_coords *= params_mask

      def mo(r):
        return mol.mo((mo_params, None), r, grids=g, weights=w) * mol.nocc

      nuclei['loc'] = atom_coords
      return energy_gs(mo, nuclei, batch, batch)

    (e_total, e_splits), grad = jax.value_and_grad(loss, has_aux=True)(_params)
    updates, opt_state = optimizer.update(grad, opt_state)
    _params = optax.apply_updates(_params, updates)
    nuclei['loc'] = _params[1]
    params = (_params[0], _)
    return params, nuclei, opt_state, (e_total, *e_splits)

  logging.info(f"Starting... Random Seed: {seed}, Batch size: {batch_size}")

  prev_loss = 0.
  start_time = time.time()
  e_train = []
  converged = False

  logging.info(f"Batch size: {batch_size}")
  logging.info(f"Total grid points: {len(mol.grids)}")

  for i in range(epoch):
    Es_batch = []
    # this can be replaced by just shifting the grids according to the new atom
    # coordinates, instead of resampling.

    params, nuclei, opt_state, Es = update(params, nuclei, opt_state, batch)
    Es_batch.append(Es)

    # retrieve all energy terms
    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
      jnp.array(Es_batch), axis=0
    )
    print(e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc)
    # track total energy for convergence check
    e_train.append(e_total)

    if (i + 1) % 1 == 0:
      atom_coords = nuclei['loc']
      logging.info(f"Iter: {i+1}/{epoch}. Ground State Energy: {e_total:.3f}.")
      logging.info(f"current coords: {nuclei['loc'].tolist()}")
      logging.info(f'{distmat(atom_coords)*0.529177249}')

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

  mol = molecule(h2_start, spin=0, level=1, basis="6-31g", mode='go')
  geo_opt(
    mol,
    epoch=1000,
    lr=1e-3,
    batch_size=1000,
    converge_threshold=1e-7,
  )
