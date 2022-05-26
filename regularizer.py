import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap


def regularizer(wave_fun, meshgrid, params, N, eps=1e-10):
  f = lambda x: wave_fun(params, x, N)
  wave_at_grid = vmap(f)(meshgrid)  # shape: (D, N)
  col = jnp.matmul(wave_at_grid.transpose(1, 0), wave_at_grid)
  I_N = jnp.eye(N)

  return jnp.sum((col - I_N)**2)


def reg_ort(params):
  return jnp.sum((params[0].transpose()@params[0] - jnp.eye(params[0].shape[0]))**2) +\
      jnp.sum((params[1].transpose()@params[0] - jnp.eye(params[1].shape[0]))**2) +\
      jnp.sum((params[0]@params[0].transpose() - jnp.eye(params[0].shape[0]))**2) +\
      jnp.sum((params[1]@params[0].transpose() - jnp.eye(params[1].shape[0]))**2)
