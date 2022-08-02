import jdft
import logging
import time
import jax
import jax.numpy as jnp
from jdft.sampler import batch_sampler
from energy import energy_gs
import optax

from pyscf import gto
from pyscf.dft import gen_grid

logging.getLogger().setLevel(logging.INFO)


def train(mol, epoch, lr, seed=123, converge_threshold=1e-3, batchsize=1000):

  params = mol._init_param(seed)
  optimizer = optax.sgd(lr)
  opt_state = optimizer.init(params)
  key = jax.random.PRNGKey(seed)

  @jax.jit
  def update(params, opt_state, grids, weights):

    def loss(params):

      def mo(r):
        return mol.mo(params, r) * mol.nocc

      return energy_gs(mo, mol.nuclei, grids, weights)

    (Egs, Es), Egs_grad = jax.value_and_grad(loss, has_aux=True)(params)
    Ek, Ee, Ex, Eh, En = Es

    updates, opt_state = optimizer.update(Egs_grad, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, Egs, Ek, Ee, Ex, Eh, En

  logging.info(f'Starting... Random Seed: {seed}, Batch size: {batchsize}')

  current_loss = 0
  batch_seeds = jnp.asarray(
    jax.random.uniform(key, (epoch,)) * 100000, dtype=jnp.int32
  )
  Egs_train = []
  Ek_train = []
  Ee_train = []
  Ex_train = []
  Eh_train = []
  En_train = []

  start_time = time.time()
  timer = []

  for i in range(epoch):

    batch_grids, batch_weights = batch_sampler(
      mol.grids, mol.weights, batchsize=batchsize, seed=batch_seeds[i]
    )
    if i == 0:
      logging.info(
        f'Batch size: {batch_grids[0].shape[0]}. \
          Number of batches in each epoch: {len(batch_grids)}'
      )

    nbatch = len(batch_grids)
    batch_tracer = jnp.zeros(6)

    for g, w in zip(batch_grids, batch_weights):
      params, opt_state, Egs, Ek, Ee, Ex, Eh, En = update(
        params, opt_state, g, w
      )
      batch_tracer += jnp.asarray([Egs, Ek, Ee, Ex, Eh, En])

    if (i + 1) % 1 == 0:
      Batch_mean = batch_tracer / nbatch
      assert Batch_mean.shape == (6,)

      Egs_train.append(Batch_mean[0].item())
      Ek_train.append(Batch_mean[1].item())
      Ee_train.append(Batch_mean[2].item())
      Ex_train.append(Batch_mean[3].item())
      Eh_train.append(Batch_mean[4].item())
      En_train.append(Batch_mean[5].item())

      print(f'Iter: {i+1}/{epoch}. Ground State Energy: {Egs_train[-1]:.3f}.')

    if jnp.abs(current_loss - Batch_mean[0].item()) < converge_threshold:
      print(
        'Converged at iteration {}. Training Time: {:.3f}s'.format(
          (i + 1),
          time.time() - start_time
        )
      )
      print('E_Ground state: ', Egs_train[-1])
      print('E_kinetic: ', Ek_train[-1])
      print('E_ext: ', Ee_train[-1])
      print('E_Hartree: ', Eh_train[-1])
      print('E_xc: ', Ex_train[-1])
      print('E_nuclear_repulsion:', En_train[-1])

      return

    else:
      current_loss = Batch_mean[0].item()

    timer.append(time.time() - start_time)

  print(
    'Not Converged. Training Time: {:.3f}s'.format(time.time() - start_time)
  )
  print('E_Ground state: ', Egs_train[-1])
  print('E_kinetic: ', Ek_train[-1])
  print('E_ext: ', Ee_train[-1])
  print('E_Hartree: ', Eh_train[-1])
  print('E_xc: ', Ex_train[-1])
  print('E_nuclear_repulsion:', En_train[-1])


if __name__ == '__main__':
  from jdft.geometries import h2o_geometry
  mol = jdft.molecule(h2o_geometry, spin=0, level=1, basis='6-31g')
  train(mol, epoch=100, lr=1e-2, batchsize=2000, converge_threshold=1e-5)