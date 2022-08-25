"""Useful functions."""

import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
# from scipy.special import factorial2 as factorial2


def euclidean_distance(x, y):
  """Euclidean distance."""
  return jnp.sqrt(jnp.sum((x - y)**2 + 1e-18))


def factorial(x):
  """Calculate the factorial of x."""
  x = jnp.asarray(x, dtype=jnp.float32)
  return jnp.exp(jax.lax.lgamma(x + 1))


def r2(x, y):
  """Square of euclidean distance."""
  return jnp.sum((x - y)**2)


def distmat(x, y=None):
  """Distance matrix."""
  if y is None:
    y = x
  return vmap(lambda x1: vmap(lambda y1: euclidean_distance(x1, y1))(y))(x)


def gaussian_intergral(alpha, n):
  r"""Compute the integral of the gaussian basis.

  Not the confuse with gaussian cdf.
  ref: https://mathworld.wolfram.com/GaussianIntegral.html
  return  \int x^n exp(-alpha x^2) dx
  """
  # if n==0:
  #     return jnp.sqrt(jnp.pi/alpha)
  # elif n==1:
  #     return 0
  # elif n==2:
  #     return 1/2/alpha * jnp.sqrt(jnp.pi/alpha)
  # elif n==3:
  #     return 0
  # elif n==4:
  #     return 3/4/alpha**2 * jnp.sqrt(jnp.pi/alpha)
  # elif n==5:
  #     return 0
  # elif n==6:
  #     return 15/8/alpha**3 * jnp.sqrt(jnp.pi/alpha)
  # elif n==7:
  #     return 0
  # else:
  #     raise NotImplementedError()

  return (
    (n == 0) * jnp.sqrt(jnp.pi / alpha) +
    (n == 2) * 1 / 2 / alpha * jnp.sqrt(jnp.pi / alpha) +
    (n == 4) * 3 / 4 / alpha**2 * jnp.sqrt(jnp.pi / alpha) +
    (n == 6) * 15 / 8 / alpha**3 * jnp.sqrt(jnp.pi / alpha)
  )


def decov(cov):
  """Decomposition of covariance matrix."""
  v, u = jnp.linalg.eigh(cov)
  v = jnp.diag(jnp.real(v)**(-1 / 2)) + jnp.eye(v.shape[0]) * 1e-7
  ut = jnp.real(u).transpose()
  return jnp.matmul(v, ut)


# def decov(cov):
#   l = jnp.linalg.cholesky(cov)
#   l = jnp.linalg.inv(l)
#   return l


def set_diag_zero(x):
  """Set diagonal items to zero."""
  return x.at[jnp.diag_indices(x.shape[0])].set(0)


def minibatch_vmap(f, in_axes=0, batch_size=10):
  """
    automatic batched vmap operation.
  """
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

    out = jax.lax.map(_process_batch, indices)
    if isinstance(out, jnp.ndarray):
      out = jnp.reshape(out, (-1, *out.shape[2:]))[:num]
    elif isinstance(out, (tuple, list)):
      out = tuple(jnp.reshape(o, (-1, *o.shape[2:]))[:num] for o in out)
    return out

  return _minibatch_vmap_f
