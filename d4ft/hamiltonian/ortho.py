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

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def sqrt_inv(mat: Float[Array, "a a"]) -> Float[Array, "a a"]:
  """Square root of inverse."""
  v, u = jnp.linalg.eigh(mat)
  v = jnp.clip(v, a_min=0)
  v = jnp.diag(jnp.real(v)**(-1 / 2))
  ut = jnp.real(u).transpose()
  return jnp.matmul(v, ut)


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
