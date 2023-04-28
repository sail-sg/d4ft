import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from d4ft.integral.obara_saika import (
  electron_repulsion_integral, kinetic_integral, nuclear_attraction_integral,
  overlap_integral
)


def get_random_gto_pairs(batch):
  key = jax.random.PRNGKey(42)
  keys = jax.random.split(key, 13)
  ra = jax.random.normal(keys[0], (batch, 3))
  rb = jax.random.normal(keys[1], (batch, 3))
  rc = jax.random.normal(keys[2], (batch, 3))
  rd = jax.random.normal(keys[3], (batch, 3))
  za = jax.random.uniform(keys[4], (batch,), minval=1., maxval=10.)
  zb = jax.random.uniform(keys[5], (batch,), minval=1., maxval=10.)
  zc = jax.random.uniform(keys[6], (batch,), minval=1., maxval=10.)
  zd = jax.random.uniform(keys[7], (batch,), minval=1., maxval=10.)
  na = jax.random.randint(keys[8], (batch, 3), minval=0, maxval=2)
  nb = jax.random.randint(keys[9], (batch, 3), minval=0, maxval=2)
  nc = jax.random.randint(keys[10], (batch, 3), minval=0, maxval=2)
  nd = jax.random.randint(keys[11], (batch, 3), minval=0, maxval=2)
  C = jax.random.normal(keys[12], (batch, 3))
  return list(
    map(np.array, [ra, rb, rc, rd, za, zb, zc, zd, na, nb, nc, nd, C])
  )


def get_obsa_intor(data_size):

  data = get_random_gto_pairs(data_size)

  @jax.jit
  def overlap():
    energies = []
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, _ in zip(*data):
      a, b = (na, ra, za), (nb, rb, zb)
      e = overlap_integral(a, b, use_horizontal=False)
      energies.append(e)
    return energies

  @jax.jit
  def kin():
    energies = []
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, _ in zip(*data):
      a, b = (na, ra, za), (nb, rb, zb)
      e = kinetic_integral(a, b)
      energies.append(e)
    return energies

  @jax.jit
  def ext():
    energies = []
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, C in zip(*data):
      a, b = (na, ra, za), (nb, rb, zb)
      e = nuclear_attraction_integral(C, a, b)
      energies.append(e)
    return energies

  @jax.jit
  def eri():
    energies = []
    for ra, rb, rc, rd, za, zb, zc, zd, na, nb, nc, nd, _ in zip(*data):
      a, b, c, d = (na, ra, za), (nb, rb, zb), (nc, rc, zc), (nd, rd, zd)
      e = electron_repulsion_integral(a, b, c, d)
      energies.append(e)
    return energies

  return dict(overlap=overlap, kin=kin, ext=ext, eri=eri)


jit_iters = 5
for data_size in range(10, 50, 10):
  fn_dict = get_obsa_intor(data_size)

  n_iters = 100

  for fn_name, fn in fn_dict.items():
    total_t = 0
    for t in range(n_iters):
      start_t = time.time()
      fn()
      iter_t = time.time() - start_t
      if t >= jit_iters:
        total_t += iter_t

    print(fn_name, data_size, total_t / (n_iters - jit_iters))
