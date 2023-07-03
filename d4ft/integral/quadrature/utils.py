# Copyright 2023 Garena Online Private Limited
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

from typing import Callable, Generator

import jax
import jax.numpy as jnp
from jaxtyping import Array


def make_quadrature_points_batches(
  grids: Array,
  weights: Array,
  batch_size: int,
  epochs: int,
  num_copies: int,
  seed: int = 137
) -> Generator:
  """
  Args:
    grids: an array of 3d electron coordinates
    weights: an array of weights for each point in the grid
  Return:
   a generator of batches
  """

  @jax.jit
  def shuffle_fn(key):
    return shuffle_quadrature_points(grids, weights, batch_size, key)

  rng = jax.random.PRNGKey(seed)
  key, rng = jax.random.split(rng)

  for _ in range(epochs):

    batches = []
    for _ in range(num_copies):
      key, rng = jax.random.split(rng)
      batches.append(shuffle_fn(key))

    for batch in zip(*batches):
      if num_copies == 1:
        yield batch[0]
      else:
        yield batch


def shuffle_quadrature_points(
  grids: Array,
  weights: Array,
  batch_size: int,
  key: Array,
):
  """Shuffle all

  Args:
    grids: an array of 3d electron coordinates
    weights: an array of weights for each point in the grid
    batch_size: batch size
    key: jax prng key
  Returns:
    shuffled quadrature points
  """
  # combine grids and weights
  weights = jnp.expand_dims(weights, 1)
  gw = jnp.concatenate((grids, weights), axis=1)

  # shuffle quadrature points
  num_total_points = grids.shape[0]
  # key = jax.random.PRNGKey(seed)
  gw = jax.random.permutation(key, gw)
  shuffled_grids = gw[:, :3]
  shuffled_weights = jnp.squeeze(gw[:, 3])

  # make batches
  batch_size = min(num_total_points, batch_size)
  num_batches = num_total_points // batch_size
  batched_g = jnp.split(shuffled_grids[:num_batches * batch_size], num_batches)
  rescaled_w = shuffled_weights * num_total_points / batch_size
  batched_w = jnp.split(rescaled_w[:num_batches * batch_size], num_batches)

  return list(zip(batched_g, batched_w))


def quadrature_integral(integrand: Callable, *coords_and_weights) -> Array:
  """Integrate a multivariate function with quadrature.

  - The integral can be computed with a randomly sampled batch of
  quadrature points, as described in the D4FT paper.

  Args:
    integrand: a multivariable function, with kwargs keepdims
    keepdims: whether to output the outer product matrix as explained above
    coords_and_weights: the quadrature points that the function will
      be evaluated and the weights of these points.

  Returns:
    A scalar value as the result of the integral.
  """
  # break down the coordinates and the weights
  coords = [coord for coord, _ in coords_and_weights]
  weights = [weight for _, weight in coords_and_weights]
  # vmap the integrand
  num_dims = len(coords_and_weights)
  fn = integrand
  for i in range(num_dims):
    in_axes = (None,) * (num_dims - i - 1) + (0,) + (None,) * (i)
    fn = jax.vmap(fn, in_axes=in_axes)
  out = fn(*coords)

  if len(weights) == 1:
    return jnp.sum((out.T * weights[0]).T, axis=0)

  # multi-dimensional quadrature
  for weight in reversed(weights):
    out = jnp.dot(out, weight)

  return out


def get_integrand(f: Callable, g: Callable, keepdims: bool = False) -> Callable:
  """
  If the integrand is the inner product of two vectors of basis functions
  F(r)=(f_1(r), ..., f_N(r)) and G(r)=(g_1(r), ..., g_N(r)), we can keep
  the basis information by computing the outer product matrix <f_i|g_j>.

  - Since the integral is computed with quadrature, i.e.

  .. math::
  <f_i|g_j>= \\sum_k w_k \\dot (f_i(r_k)g_j(r_k))

  where w_k is the quadrature weight and r_k are the grid points,
  the outer prodcut matrix can be computed as the outer product of the
  vector F(r) and G(r):

  .. math::
  (<f_i|g_j>)_{ij} = \\sum_k w_k \\dot (F(r_k)G(r_k)^T)
  """

  def integrand(r: Array) -> Array:
    F, G = f(r), g(r)
    if keepdims:
      if len(F.shape) == 1:
        return jnp.outer(F, G)
      else:  # map over spin axis
        return jax.vmap(jnp.outer)(F, G)
    return jnp.sum(F * G)

  return integrand


def wave2density(orbitals: Callable, polarized=False) -> Callable:
  """
  Transform the wave function into density function.
  Args:
    mo: a [3] -> [2, N] function, where N is the number of molecular orbitals.
      mo only takes one argment, which is the coordinate.
    polarized: if True will return a 1D array with two elements indicating
      the density of each spin
  Return:
    density function: [3] -> float or 1D array.
  """
  if polarized:
    return lambda r: jnp.sum((orbitals(r))**2, axis=1)
  else:
    return lambda r: jnp.sum((orbitals(r))**2)
