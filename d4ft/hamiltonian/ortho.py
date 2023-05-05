"""Functions for getting parametrized orthogonal matrices"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array


def qr_factor(
  params: Array,
  batch_dim: Optional[int] = None,
  row_wise: bool = True
) -> Array:
  """Get a orthongal matrix parametrized with qr factor.

  Args:
    batch_dim: if provided vmap over this dim
    row_wise: if true return row-wise orthogonal matrix. For non-square matrix
  of size (a,b) where a<b, QR returns orthogonal column vectors of shape (a,a).
  To get row-wise orthogonality transposition is needed.
  """
  qr_fn = lambda p: jnp.linalg.qr(p)[0]
  if batch_dim:
    qr_fn = jax.vmap(qr_fn, batch_dim, batch_dim)
  orthogonal = qr_fn(params)
  if row_wise:
    transpose_axis = (0,) + tuple(range(1, len(params.shape)))
    orthogonal = jnp.transpose(orthogonal, transpose_axis)
  return orthogonal
