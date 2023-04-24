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
"""Useful functions."""

from functools import partial
from itertools import chain, combinations
from math import gcd
from typing import Callable, List

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap


def wave2density(mo: Callable, nocc=1., keep_spin=False):
  """
  Transform the wave function into density function.
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
        mo only takes one argment, which is the coordinate.
    keep_spin: if True will return a 1D array with two elements indicating
    the density of each spin
  Return:
    density function: [3] -> float or 1D array.
  """

  if keep_spin:
    return lambda r: jnp.sum((mo(r) * nocc)**2, axis=1)
  else:
    return lambda r: jnp.sum((mo(r) * nocc)**2)


def euclidean_distance(x, y):
  """Euclidean distance."""
  return jnp.sqrt(jnp.sum((x - y)**2 + 1e-18))


def factorial(x):
  """Calculate the factorial of x."""
  x = jnp.asarray(x, dtype=jnp.float32)
  return jnp.exp(jax.lax.lgamma(x + 1))


def distmat(x, y=None):
  """Distance matrix."""
  if y is None:
    y = x
  return vmap(lambda x1: vmap(lambda y1: euclidean_distance(x1, y1))(y))(x)


def decov(cov):
  """Decomposition of covariance matrix."""
  v, u = jnp.linalg.eigh(cov)
  v = jnp.clip(v, a_min=0)
  v = jnp.diag(jnp.real(v)**(-1 / 2))
  ut = jnp.real(u).transpose()
  return jnp.matmul(v, ut)


def set_diag_zero(x):
  """Set diagonal items to zero."""
  return x.at[jnp.diag_indices(x.shape[0])].set(0)


def factorization(n):

  factors = []

  def get_factor(n):
    x_fixed = 2
    cycle_size = 2
    x = 2
    factor = 1

    while factor == 1:
      for _ in range(cycle_size):
        if factor > 1:
          break
        x = (x * x + 1) % n
        factor = gcd(x - x_fixed, n)

      cycle_size *= 2
      x_fixed = x

    return factor

  while n > 1:
    next = get_factor(n)
    factors.append(next)
    n //= next

  return factors


def powerset(iterable):
  s = list(iterable)
  return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_exact_batch(factors: List[int], batch_size_upperbound: int):
  """
  Args:
    batch_size: the target upperbound. If number of ijlk index is lower than
      this, then use full batch, otherwise find the closest divider of dataset
      size lower than this upperbound.
  """
  batch_sizes = list(map(np.prod, powerset(factors)))
  dists = list(map(lambda b: np.abs(b - batch_size_upperbound), batch_sizes))
  return int(batch_sizes[np.argmin(dists)])


def minibatch_vmap(
  f: Callable,
  full_batch_size: int,
  in_axes=0,
  batch_size: int = 1000,
  exact_division: bool = False,
  reduce: bool = False,
  thresh: int = 10000,  # TODO: tune this
):
  """
  TODO: speed test this

    automatic batched vmap operation.

  reduce only support sum now
  """

  logging.info(f"full_batch_size: {full_batch_size}")

  batch_f = jax.vmap(f, in_axes=in_axes)
  if full_batch_size <= batch_size or full_batch_size <= thresh:
    if not reduce:
      return batch_f
    else:

      def reduced_batch_f(*args):
        return jnp.sum(batch_f(*args))

      return reduced_batch_f

  def _minibatch_vmap_f(*args):
    nonlocal in_axes, batch_size
    if not isinstance(in_axes, (tuple, list)):
      in_axes = (in_axes,) * len(args)
    for i, ax in enumerate(in_axes):
      if ax is not None:
        if args[i][0].shape == ():
          num = len(args[i])
        elif isinstance(args[i], tuple):  # use the first element
          num = args[i][0].shape[ax]
        else:
          num = args[i].shape[ax]
    if exact_division:
      factors = sorted(factorization(num))
      exact_batch_size = get_exact_batch(factors, batch_size)
      if exact_batch_size == 1:  # num is prime
        num_shards = int(np.floor(num / batch_size))
      else:
        batch_size = exact_batch_size
        num_shards = num // batch_size
    else:
      num_shards = int(np.floor(num / batch_size))
    size = num_shards * batch_size
    remainder = num % batch_size
    indices = jnp.arange(0, size, batch_size)

    logging.info(
      f"batch_size: {batch_size}, num_shards: {num_shards}, "
      f"remainder: {remainder}"
    )

    def _process_batch(start_index, batch_size):

      def slice_arg_i(arg, ax):
        return jax.lax.dynamic_slice_in_dim(
          arg,
          start_index=start_index,
          slice_size=batch_size,
          axis=ax,
        )

      def slice_arg(arg, ax):
        """handles tuple input"""
        if isinstance(arg, tuple):
          return [slice_arg_i(arg_i, ax) for arg_i in arg]
        else:
          return slice_arg_i(arg, ax)

      batch_args = (
        slice_arg(arg, ax) if ax is not None else arg
        for arg, ax in zip(args, in_axes)
      )
      out = batch_f(*batch_args)

      if not reduce:
        return out
      else:
        return jnp.sum(out)

    out = jax.lax.map(partial(_process_batch, batch_size=batch_size), indices)
    out_remainder = _process_batch(size, remainder)
    if not reduce:
      if isinstance(out, jnp.ndarray):
        out = jnp.reshape(out, (-1, *out.shape[2:]))[:num]
      elif isinstance(out, (tuple, list)):
        out = tuple(jnp.reshape(o, (-1, *o.shape[2:]))[:num] for o in out)
      out = jnp.concatenate([out, out_remainder])
    else:
      out = jnp.sum(out)
      out += out_remainder
    return out

  return _minibatch_vmap_f
