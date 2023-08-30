# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for getting parametrized orthogonal matrices"""

from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def sqrt_inv_eig(A: Float[Array, "a a"]) -> Float[Array, "a a"]:
  """Compute square root of inverse with eigen-decomposition."""
  v, u = jnp.linalg.eigh(A)
  v = (v + jnp.abs(v)) / 2
  v_sqrt = jnp.diag(v**(-1 / 2))
  return v_sqrt @ u.T


def sqrt_inv_cholesky(A: Float[Array, "a a"]) -> Float[Array, "a a"]:
  """Square root of inverse."""
  A_inv = jnp.linalg.inv(A)
  L = jnp.linalg.cholesky(A_inv)
  return L.T


def sqrt_inv_svd(A: Float[Array, "a a"]) -> Float[Array, "a a"]:
  """Square root of inverse."""
  u, s, _ = jnp.linalg.svd(A)
  s = (s + jnp.abs(s)) / 2
  s_sqrt = jnp.diag(s**(-1 / 2))
  return s_sqrt @ u.T


def sqrt_inv(
  A: Float[Array, "a a"],
  method: Literal["eig", "cholesky", "svd"] = "cholesky"
) -> Float[Array, "a a"]:
  """Square root of inverse."""
  if method == "eig":
    return sqrt_inv_eig(A)
  elif method == "cholesky":
    return sqrt_inv_cholesky(A)
  elif method == "svd":
    return sqrt_inv_svd(A)
  else:
    raise NotImplementedError(f"method {method} not implemented")


def qr_factor(
  params: Array,
  batch_dim: Optional[int] = None,
  row_wise: bool = True
) -> Array:
  """Get a orthongal matrix parametrized with qr factor.

  NOTE: QR decomposition is done column-wise, and is only unique up
  to a column-wise phase shift. In the case of real coefficients, this
  means given an input matrix :math:`A`, the output of QR decomposition
  :math:`A=QR` is only unique up to a sign flip of each column of :math:`Q`.
  In D4FT the convention for MO coefficients is that each row represents a MO,
  so row_wise must be true.

  Another case for setting row_wise is even when the input is non-square.
  For non-square matrix of size (a,b) where a<b, QR returns orthogonal column
  vectors of shape (a,a). To get row-wise orthogonality transposition is needed:
  first transpose the input matrix to (b,a), then QR returns orthogonal columns
  of size (b,a), which are rows in the original space.

  Args:
    batch_dim: if provided vmap over this dim
    row_wise: if true return row-wise orthogonal matrix.
  """
  qr_fn = lambda p: jnp.linalg.qr(p)[0]
  if batch_dim:
    qr_fn = jax.vmap(qr_fn, batch_dim, batch_dim)
  orthogonal = qr_fn(params)
  if row_wise:
    transpose_axis = (0,) + tuple(reversed(range(1, len(params.shape))))
    orthogonal = jnp.transpose(orthogonal, transpose_axis)
  return orthogonal
