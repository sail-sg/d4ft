# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging
import time
import jax
import jax.numpy as jnp
from d4ft.energy import calc_energy
import optax
from d4ft.grids import _gen_grid
# from d4ft.functions import distmat
import copy
from pyscf import gto
from pyscf.dft import gen_grid

from jax.config import config

config.update("jax_debug_nans", True)


def _get_config_from_coords(nuclei):
  output = []
  natm = len(nuclei['symbol'])
  output += [
    nuclei['symbol'][i] + ' ' +
    ' '.join(str(j * 0.529177249)
             for j in nuclei['loc'][i].tolist()) + ';'
    for i in range(natm)
  ]
  output = ' '.join(i for i in output)
  return output


def geo_opt(mol, epoch, lr, seed=123, converge_threshold=1e-3, geo_mask=None):
  """Run the main training loop."""

  nuclei = mol.nuclei

  bond_length = jnp.zeros([1, 1]) + nuclei['loc'][-1, -1]

  params = (*mol._init_param(seed), bond_length)
  print(params)

  schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.00001,
    peak_value=0.01,
    warmup_steps=500,
    decay_steps=epoch / 2,
    end_value=lr,
  )

  optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=schedule),
  )

  # optimizer = optax.adam(lr)
  opt_state = optimizer.init(params)

  _grids, _weights, _atoms = _gen_grid(mol.pyscf_mol, level=1, atom_label=True)

  @jax.jit
  def update(params, nuclei, opt_state, batch):  # params only contains mo.
    g, w = batch

    def loss(params):
      mo_params, ao_params, bond_length = params
      atom_coords = nuclei['loc'] * (1 - geo_mask) + geo_mask * bond_length

      # _grids = _grid_shift(g, _atoms, nuclei['loc'], atom_coords)

      def mo(r):
        return mol.mo(
          (mo_params, ao_params),
          r,
          grids=g,
          weights=w,
          atom_coords=atom_coords
        ) * mol.nocc

      nuclei_new = {'loc': atom_coords, 'charge': nuclei['charge']}
      # nuclei_new = nuclei

      e_total, e_splits = calc_energy(mo, nuclei_new, batch, batch)
      return e_total, (e_splits, nuclei_new)

    (e_total, aux), grad = jax.value_and_grad(
      loss, argnums=0, has_aux=True
    )(
      params
    )

    e_splits, nuclei_new = aux
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    # nuclei = {'loc': momentum * nuclei['loc'] + (1-momentum) * _params[1],
    # 'charge': nuclei['charge']}
    # nuclei['loc'] = momentum * nuclei['loc'] + (1-momentum) * _params[1]
    return params, nuclei_new, opt_state, (e_total, *e_splits)

  logging.info(f"Starting... Random Seed: {seed}")

  prev_loss = 0.
  start_time = time.time()
  e_train = []
  converged = False
  nuclei_old = copy.deepcopy(nuclei)
  nuclei_new = copy.deepcopy(nuclei)

  logging.info(f"Total grid points: {len(mol.grids)}")

  for i in range(epoch):
    Es_batch = []

    # _grids, _weights = _gen_grid(
    #   mol.pyscf_mol, level=mol.level, atom_label=False,
    # atom_coords=nuclei['loc']
    # )

    # this can be replaced by just shifting the grids according to the new atom
    # coordinates, instead of resampling.

    # _grids = _grid_shift(_grids, _atoms, nuclei_old['loc'], nuclei_new['loc'])
    # mol.atom_coords
    config = _get_config_from_coords(
      {
        'symbol': mol.atom_symbol,
        'loc': nuclei_old['loc'],
        'charge': nuclei_old['charge']
      }
    )

    print(config)
    _mol = gto.M(atom=config, basis=mol.basis, spin=0)
    g = gen_grid.Grids(_mol)
    g.level = mol.level
    g.build()
    _grids = jnp.array(g.coords)
    _weights = jnp.array(g.weights)

    batch = (_grids, _weights)
    # batches = batch_sampler(_grids, _weights, batch_size, seed=seed+i)
    # batch = (mol.grids, mol.weights)

    # for batch in batches:
    nuclei_old = copy.deepcopy(nuclei_new)
    params, nuclei_new, opt_state, Es = update(
      params, nuclei_old, opt_state, batch
    )
    Es_batch.append(Es)
    # mol.mo.ao.atom_coords = nuclei_new['loc']
    # retrieve all energy terms
    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
      jnp.array(Es_batch), axis=0
    )

    if jnp.isnan(e_total):
      print(params)
      print(nuclei_old)
      print(nuclei_new)
      print(e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc)
      print(_grids)
      break
    # track total energy for convergence check
    e_train.append(e_total)

    if (i + 1) % 10 == 0:
      logging.info("Energy:")
      logging.info(f" Ground State: {e_total}")
      logging.info(f" Kinetic: {e_kin}")
      logging.info(f" External: {e_ext}")
      logging.info(f" Exchange-Correlation: {e_xc}")
      logging.info(f" Hartree: {e_hartree}")
      logging.info(f" Nucleus Repulsion: {e_nuc}")
      logging.info(f"Iter: {i+1}/{epoch}. Ground State Energy: {e_total:.3f}.")
      # logging.info(f"current coords: {nuclei_new['loc'].tolist()}")
      # logging.info(f"{distmat(nuclei_new['loc'])*0.529177249}")

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
  from d4ft.geometries import h2_geometry
  from Molecule import Molecule
  print(h2_geometry)

  h2_start = """
  H 0.0000 0.0000 0.0000;
  H 0.0000 0.0000 1.2;
  """
  print(h2_start)

  mol = Molecule(h2_start, spin=0, level=3, basis="6-31g", mode='go')

  # print(mol.nuclei)
  # print(_get_config_from_coords(mol.nuclei))
  geo_mask = jnp.ones_like(mol.nuclei['loc'])
  geo_mask = geo_mask.at[:, 0].set(0)
  geo_mask = geo_mask.at[:, 1].set(0)
  geo_mask = geo_mask.at[0, :].set(0)

  geo_opt(
    mol,
    epoch=5000,
    lr=1e-3,
    seed=1235,
    batch_size=50000,
    converge_threshold=1e-8,
    geo_mask=geo_mask
  )
