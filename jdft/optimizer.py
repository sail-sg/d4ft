from copy import deepcopy
import logging
import time
import jax
import jax.numpy as jnp
import optax
from jdft.intor import Quadrature
from jdft.energy import E_gs
from jdft.sampler import batch_sampler
from jdft.visualization import save_contour

def sgd(
    mol,
    epoch,
    lr=1e-3,
    seed=123,
    converge_threshold=1e-3,
    batchsize=1000,
    save_fig=False,
    **args
  ):
  """
  Calculate the ground state wave functions.
  Args:
    mol: jdft molecule object
    epoch: number of epochs
    lr: learning rate
    seed: random seed
    converge_threshold:
    batchsize: the number of grids sampled for each batch
  Return:
    Egs: ground state energy
    Params: converged parameters for the mol objects.
  """

  if mol.params is None:
    mol.params = mol._init_param(seed)

  params = deepcopy(mol.params)
  # schedule = optax.warmup_cosine_decay_schedule(
  #             init_value=0.5,
  #             peak_value=1,
  #             warmup_steps=50,
  #             decay_steps=500,
  #             end_value=lr,
  #             )

  if 'optimizer' in args:
    if args['optimizer'] == 'sgd':
      optimizer = optax.sgd(lr)
      # optimizer = optax.chain(
      #     optax.clip(1.0),
      #     optax.sgd(learning_rate=schedule),
      #     )

    elif args['optimizer'] == 'adam':
      optimizer = optax.adam(lr)
      # optimizer = optax.chain(
      #     optax.clip(1.0),
      #     optax.adam(learning_rate=schedule),
      #     )
    else:
      raise NotImplementedError('Optimizer in [\'sgd\', \'adam\']')
  else:
    optimizer = optax.sgd(lr)

  opt_state = optimizer.init(params)
  key = jax.random.PRNGKey(seed)

  @jax.jit
  def update(params, opt_state, grids, weights, *args):

    def loss(params):
      intor = Quadrature.from_mo(mol.mo, mol.nocc, params, grids, weights)
      return E_gs(intor, mol.nuclei)

    (Egs, Es), Egs_grad = jax.value_and_grad(loss, has_aux=True)(params)
    Ek, Ee, Ex, Eh, En = Es

    mol.mo.params = params
    params, opt_state = mol.mo.update(Egs_grad, optimizer, opt_state, *args)

    return params, opt_state, Egs, Ek, Ee, Ex, Eh, En

  if save_fig:
    file = '/home/aiops/litb/project/dft/experiment/figure/{0:04}'.format(
      0
    ) + '.png'
    save_contour(mol, file)

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
  mol.timer = []

  for i in range(epoch):

    batch_grids, batch_weights = batch_sampler(
      mol.grids, mol.weights, batchsize=batchsize, seed=batch_seeds[i]
    )
    if i == 0:
      logging.info(
        f'Batch size: {batch_grids[0].shape[0]}. Number of batches in each epoch: {len(batch_grids)}'
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

      if save_fig:
        file = '/home/aiops/litb/project/dft/experiment/figure/{0:04}'.format(
          i + 1
        ) + '.png'
        save_contour(mol, file)

    if jnp.abs(current_loss - Batch_mean[0].item()) < converge_threshold:
      mol.params = deepcopy(params)
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
      mol.tracer += Egs_train

      return

    else:
      current_loss = Batch_mean[0].item()

    mol.timer.append(time.time() - start_time)

  mol.tracer += Egs_train
  mol.params = deepcopy(params)
  print(
    'Not Converged. Training Time: {:.3f}s'.format(time.time() - start_time)
  )
  print('E_Ground state: ', Egs_train[-1])
  print('E_kinetic: ', Ek_train[-1])
  print('E_ext: ', Ee_train[-1])
  print('E_Hartree: ', Eh_train[-1])
  print('E_xc: ', Ex_train[-1])
  print('E_nuclear_repulsion:', En_train[-1])
  return Egs_train[-1]
