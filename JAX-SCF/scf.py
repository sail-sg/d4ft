import jax
import jax.numpy as jnp
from energy import energy_gs, wave2density, integrand_kinetic, integrand_external, integrand_xc_lda, e_nuclear
from typing import Callable
from absl import logging
import tensorflow as tf
import numpy as np
import time

def hamil_kinetic(ao: Callable, batch):
  """
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
  """
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
      nuclei_charge / jnp.sqrt(jnp.sum((r - nuclei_loc)**2, axis=1) + 1e-16)
    )

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: v(r) * m(r), batch)


def energy_external(mo: Callable, nuclei, batch):
  return integrate_s(integrand_external(mo, nuclei), batch)


"""def hamil_hartree(ao: Callable, mo_old, batch1, batch2):
  density = wave2density(mo_old)

  def g(r):

    def v(x):
      return density(x) / jnp.clip(
        jnp.linalg.norm(x - r), a_min=1e-8
      ) * jnp.any(x != r)

    return integrate_s(v, batch1)

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: g(r) * m(r), batch2)
"""


def hamil_hartree(ao: Callable, mo_old, batch):
  density = wave2density(mo_old)

  def g(r):
    r"""
      g(r) = \int n(r')/|r-r'| dr'
    """

    def v(x):
      return density(x) / jnp.clip(
        jnp.linalg.norm(x - r), a_min=1e-8
      ) * jnp.any(x != r)

    return integrate_s(v, batch)

  def m(r):
    return jax.vmap(jnp.outer)(ao(r), ao(r))

  return integrate_s(lambda r: g(r) * m(r), batch)


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
      return density(x) / jnp.clip(
        jnp.linalg.norm(x - r), a_min=1e-8
      ) * jnp.any(x != r)

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


def get_fork(ao: Callable, mo_old: Callable, nuclei, batch):
  """H_kin, H_ext, H_lda, H_hartree = 0, 0, 0, 0
  for batch in dataset1.as_numpy_iterator():
    H_kin, H_ext, H_lda = H_kin + hamil_kinetic(ao, batch), H_ext + hamil_external(ao, nuclei, batch), H_lda + hamil_lda(ao, mo_old, batch)
  for batch1 in dataset1.as_numpy_iterator():
    for batch2 in dataset2.as_numpy_iterator():
      H_hartree += hamil_hartree(ao, mo_old, batch1, batch2)
  return H_kin + H_ext + H_lda + H_hartree"""
  return hamil_kinetic(ao, batch) + \
      hamil_external(ao, nuclei, batch) + \
      hamil_hartree(ao, mo_old, batch) + \
      hamil_lda(ao, mo_old, batch)


def get_energy(mo_old: Callable, nuclei, batch):
  e_kin = energy_kinetic(mo_old, batch)
  e_ext = energy_external(mo_old, nuclei, batch)
  e_hartree = energy_hartree(mo_old, batch)
  e_xc = energy_lda(mo_old, batch)
  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc

  return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)


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
  """def f(g, w):
    if args.mini_batch == False:
      return jnp.sum(jax.vmap(v)(g, w), axis=0)
    else:
      return jnp.sum(minibatch_vmap(v, batch_size=args.batch_size)(g, w), axis=0)"""

  return f(g, w) if args.mini_batch == False else minibatch_f(g, w)


"""def minibatch_vmap(f, in_axes=0, batch_size=10):
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
      return jnp.sum(_process_batch(start_index))

    out = jax.lax.map(_process_batch, indices)
    if isinstance(out, jnp.ndarray):
      out = jnp.reshape(out, (-1, *out.shape[2:]))[:num]
    elif isinstance(out, (tuple, list)):
      out = tuple(jnp.reshape(o, (-1, *o.shape[2:]))[:num] for o in out)
    return out

  return _minibatch_vmap_f"""


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


def scf(iter, mol, seed=123, momentum=0.5):
  batch = (mol.grids, mol.weights)
  params = mol._init_param(seed)
  mo_params, _ = params
  _diag_one_ = jnp.ones([2, mol.mo.nmo])
  _diag_one_ = jax.vmap(jnp.diag)(_diag_one_)
  
  shift = jnp.zeros(mol.mo.nmo)
  for i in range(args.shift, mol.mo.nmo):
    shift = shift.at[i].set(1)
  shift = jnp.diag(shift)
  #print(shift)

  @jax.jit
  def update(mo_params):

    def ao(r):
      return mol.mo((_diag_one_, None), r)

    def mo_old(r):
      return mol.mo((mo_params, None), r) * mol.nocc

    fork = get_fork(ao, mo_old, mol.nuclei, batch)

    fork = fork - args.sigma * shift# (mo_params @ jnp.transpose(mo_params, (0, 2, 1)))
    _, mo_params = jnp.linalg.eigh(fork)
    mo_params = jnp.transpose(mo_params, (0, 2, 1))

    def mo(r):
      return mol.mo((mo_params, None), r) * mol.nocc

    return mo_params, get_energy(mo, mol.nuclei, batch)#energy_gs(mo, mol.nuclei, batch, batch)
    #get_energy(mo, mol.nuclei, batch)
  
  @jax.jit
  def PySCF_Trick(new_params):
    def cov(cov):
      """Decomposition of covariance matrix."""
      v, u = jnp.linalg.eigh(cov)
      v = jnp.diag(jnp.real(v)**(1 / 2)) + jnp.eye(v.shape[0]) * 1e-10
      ut = jnp.real(u)
      return jnp.matmul(ut, v)

    from jdft.functions import decov
    new_params = new_params @ decov(mol.ao.overlap)
    idx = jnp.argmax(jnp.abs(jnp.real(new_params)), axis=0)
    new_params[:, jnp.real(new_params[idx, jnp.arange(jnp.shape(new_params)[-1])]) < 0] *= -1
    new_params = new_params @ cov(mol.ao.overlap)

    return new_params

  # the main loop.
  logging.info(f" Starting...SCF loop")
  for i in range(iter):
    new_params, Es = update(mo_params)
    mo_params = (1 - momentum) * new_params + momentum * mo_params
    e_total, e_splits = Es
    e_kin, e_ext, e_xc, e_hartree, e_nuc = e_splits

    logging.info(f" Iter: {i+1}/{iter}")
    logging.info(f" Ground State: {e_total}")
    logging.info(f" Kinetic: {e_kin}")
    logging.info(f" External: {e_ext}")
    logging.info(f" Exchange-Correlation: {e_xc}")
    logging.info(f" Hartree: {e_hartree}")
    logging.info(f" Nucleus Repulsion: {e_nuc}")

  """batch = (mol.grids, mol.weights)

  if batch_size > mol.grids.shape[0]:
    batch_size = mol.grids.shape[0]

  dataset1 = tf.data.Dataset.from_tensor_slices((
      mol.grids,
      mol.weights,
  )).shuffle(
      len(mol.grids), seed=seed
  ).batch(
      batch_size, drop_remainder=False
  )

  dataset2 = tf.data.Dataset.from_tensor_slices((
      mol.grids,
      mol.weights,
  )).shuffle(
      len(mol.grids), seed=seed + 1
  ).batch(
      batch_size, drop_remainder=False
  )

  params = mol._init_param(seed)
  mo_params, _ = params
  _diag_one_ = jnp.ones([2, mol.mo.nmo])
  _diag_one_ = jax.vmap(jnp.diag)(_diag_one_)

  def scf_energy_gs(mo, mol):
    dataset1_ = tf.data.Dataset.from_tensor_slices((
        mol.grids,
        mol.weights,
    )).shuffle(
        len(mol.grids), seed=seed
    ).batch(
        batch_size, drop_remainder=True
    )

    dataset2_ = tf.data.Dataset.from_tensor_slices((
        mol.grids,
        mol.weights,
    )).shuffle(
        len(mol.grids), seed=seed + 1
    ).batch(
        batch_size, drop_remainder=True
    )

    def reweigt(batch):
      g, w = batch
      w = w * len(mol.grids) / w.shape[0]
      return g, w

    Es_batch = []

    for batch1, batch2 in zip(
        dataset1_.as_numpy_iterator(), dataset2_.as_numpy_iterator()
    ):
      batch1 = reweigt(batch1) # (5000, 3), (5000, )
      batch2 = reweigt(batch2)
      e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc) = energy_gs(mo, mol.nuclei, batch1, batch2)
      Es_batch.append((e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc))

    # retrieve all energy terms
    e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc = jnp.mean(
        jnp.array(Es_batch), axis=0
    )
    return e_total, (e_kin, e_ext, e_xc, e_hartree, e_nuc)
    return 0, (0, 0, 0, 0, 0)

  #@jax.jit
  def update(mo_params, dataset1, dataset2):

    def ao(r):
      return mol.mo((_diag_one_, None), r)

    def mo_old(r):
      return mol.mo((mo_params, None), r) * mol.nocc

    fork = get_fork(ao, mo_old, mol.nuclei, dataset1, dataset2)
    _, mo_params = jnp.linalg.eigh(fork)
    mo_params = jnp.transpose(mo_params, (0, 2, 1))

    def mo(r):
      return mol.mo((mo_params, None), r) * mol.nocc
    
    return mo_params, scf_energy_gs(mo, mol) #energy_gs(mo, mol.nuclei, batch, batch)

  # the main loop.
  logging.info(f" Starting...SCF loop")
  for i in range(iter):
    new_params, Es = update(mo_params, dataset1, dataset2)
    mo_params = (1 - momentum) * new_params + momentum * mo_params
    e_total, e_splits = Es
    e_kin, e_ext, e_xc, e_hartree, e_nuc = e_splits

    logging.info(f" Iter: {i+1}/{iter}")
    logging.info(f" Ground State: {e_total}")
    logging.info(f" Kinetic: {e_kin}")
    logging.info(f" External: {e_ext}")
    logging.info(f" Exchange-Correlation: {e_xc}")
    logging.info(f" Hartree: {e_hartree}")
    logging.info(f" Nucleus Repulsion: {e_nuc}")"""


if __name__ == '__main__':
  import jdft.geometries
  from jdft.geometries import h2_geometry, o2_geometry, h2o_geometry, benzene_geometry
  from molecule import molecule
  import argparse

  parser = argparse.ArgumentParser(description="JDFT Project")
  parser.add_argument("--batch_size", type=int, default=5000)
  parser.add_argument("--mini_batch", type=bool, default=False)
  parser.add_argument("--epoch", type=int, default=5)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--geometry", type=str, default="benzene")
  parser.add_argument("--basis_set", type=str, default="6-31g")
  parser.add_argument("--momentum", type=float, default=0)
  parser.add_argument("--fock_momentum", type=float, default=0)
  parser.add_argument("--sigma", type=float, default=0)
  parser.add_argument("--shift", type=int, default=33)
  parser.add_argument("--pyscf_trick", type=bool, default=False)
  args = parser.parse_args()

  if args.geometry == "h2":
    geometry = h2_geometry
  elif args.geometry == "h2o":
    geometry = h2o_geometry
  elif args.geometry == "benzene":
    geometry = benzene_geometry

  geometry = getattr(jdft.geometries, args.geometry+"_geometry")

  mol = molecule(
    geometry, spin=0, level=1, mode="scf", basis=args.basis_set
  )
  
  start = time.time()
  scf(args.epoch, mol, seed=args.seed, momentum=args.momentum)
  end = time.time()
  logging.info(f" Time spent: {end - start}")
  logging.info(args)


"""
==========================================================================
INFO:absl: Iter: 1000/1000
INFO:absl: Ground State: -75.52349853515625
INFO:absl: Kinetic: 76.12904357910156
INFO:absl: External: -199.23477172851562
INFO:absl: Exchange-Correlation: -8.134370803833008
INFO:absl: Hartree: 46.52707290649414
INFO:absl: Nucleus Repulsion: 9.189533233642578
INFO:absl: Time spent: 26.43065881729126
INFO:absl:Namespace(batch_size=10000, epoch=1000, geometry='h2o', mini_batch=False, momentum=0.9, seed=1234)
PySCF: -75.154
==========================================================================
INFO:absl: Ground State: -1.0443077087402344
INFO:absl: Kinetic: 1.1027828454971313
INFO:absl: External: -3.5872859954833984
INFO:absl: Exchange-Correlation: -0.5561888813972473
INFO:absl: Hartree: 1.2826303243637085
INFO:absl: Nucleus Repulsion: 0.7137539982795715
INFO:absl: Time spent: 8.29686689376831
INFO:absl:Namespace(batch_size=10000, epoch=10, geometry='h2', mini_batch=False, momentum=0.0, seed=1234)
PySCF: -1.039
==========================================================================
INFO:absl: Ground State: -147.73880004882812
INFO:absl: Kinetic: 159.47427368164062
INFO:absl: External: -428.2807312011719
INFO:absl: Exchange-Correlation: -15.971941947937012
INFO:absl: Hartree: 108.99212646484375
INFO:absl: Nucleus Repulsion: 28.047487258911133
INFO:absl: Time spent: 75.38451290130615
INFO:absl:Namespace(batch_size=1000, epoch=10000, geometry='o2', mini_batch=False, momentum=0.995, seed=1234)
PySCF: -148.022
==========================================================================
INFO:absl: Ground State: -184.05947875976562
INFO:absl: Kinetic: 204.23348999023438
INFO:absl: External: -588.9234619140625
INFO:absl: Exchange-Correlation: -21.714208602905273
INFO:absl: Hartree: 164.05824279785156
INFO:absl: Nucleus Repulsion: 58.28645324707031
INFO:absl: Time spent: 50.469801902770996
INFO:absl:Namespace(batch_size=1000, epoch=577, geometry='co2', mini_batch=False, momentum=0.9, seed=1234)
PySCF: -185.58
==========================================================================
INFO:absl: Iter: 100/100
INFO:absl: Ground State: -39.229278564453125
INFO:absl: Kinetic: 39.75514221191406
INFO:absl: External: -119.94140625
INFO:absl: Exchange-Correlation: -5.968348503112793
INFO:absl: Hartree: 33.45330047607422
INFO:absl: Nucleus Repulsion: 13.47203254699707
INFO:absl: Time spent: 15.56773042678833
INFO:absl:Namespace(basis_set='sto-3g', batch_size=1000, epoch=100, geometry='ch4', mini_batch=False, momentum=0.9, seed=1234, sigma=0.0)
PySCF: -39.7268
==========================================================================
INFO:absl: Iter: 1/1
INFO:absl: Ground State: -224.12966918945312
INFO:absl: Kinetic: 241.15530395507812
INFO:absl: External: -962.1401977539062
INFO:absl: Exchange-Correlation: -30.910701751708984
INFO:absl: Hartree: 324.53936767578125
INFO:absl: Nucleus Repulsion: 203.22653198242188
INFO:absl: Time spent: 153.5192084312439
INFO:absl:Namespace(basis_set='6-31g', batch_size=2000, epoch=1, geometry='benzene', mini_batch=False, momentum=0.9, pyscf_trick=False, seed=1234, shift=55, sigma=6.0)
PySCF: -224.573862
==========================================================================
INFO:absl: Iter: 10/10
INFO:absl: Ground State: -151.28616333007812
INFO:absl: Kinetic: 164.54637145996094
INFO:absl: External: -544.6715087890625
INFO:absl: Exchange-Correlation: -19.800884246826172
INFO:absl: Hartree: 166.62908935546875
INFO:absl: Nucleus Repulsion: 82.0107421875
INFO:absl: Time spent: 77.69645428657532
INFO:absl:Namespace(basis_set='6-31g', batch_size=2000, epoch=10, geometry='ethonal', mini_batch=False, momentum=0.9, pyscf_trick=False, seed=1234, shift=30, sigma=6.0)
PySCF: -150.023
==========================================================================
INFO:absl: Iter: 1/1
INFO:absl: Ground State: -738.982666015625
INFO:absl: Kinetic: 792.5358276367188
INFO:absl: External: -4800.94384765625
INFO:absl: Exchange-Correlation: -97.14138793945312
INFO:absl: Hartree: 1883.203125
INFO:absl: Nucleus Repulsion: 1483.363525390625
INFO:absl: Time spent: 303.60129618644714
INFO:absl:Namespace(basis_set='6-31g', batch_size=10000, epoch=1, geometry='c20', mini_batch=True, momentum=0.99, pyscf_trick=False, seed=1234, shift=160, sigma=10.0)
PySCF: -737.140
==========================================================================

PySCF: -1345.433873991594
"""
