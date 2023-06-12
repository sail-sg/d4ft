import jax.numpy as jnp
from jaxtyping import Array, Float


def kinetic_integral(gvec, kpts, _params_w, nocc) -> Float[Array, ""]:
  """Kinetic energy
      E = 1/2 \sum_{G} |k + G|^2 c_{i,k,G}^2
  Args:
      gvec (4D array): G-vector. Shape: [N1, N2, N3, 3]
      kpts (2D array): k-points. Shape: [nk, 3]
      _params_w (6D array): whitened model params. Shape: [2, ni, nk, M1, M2, M3]
      nocc (3D array): occupation mask. Shape: [2, ni, nk]
  Returns:
      scalar
  """
  _g = gvec[None, None, :, :, :, :]  # shape [1, 1, M1, M2, M3, 3]
  _k = kpts[None, :, None, None, None, :]  # shape [1, nk, 1, 1, 1, 3]

  output = jnp.sum((_g + _k)**2, axis=-1)  # [1, nk, M1, M2, M3]
  output = jnp.expand_dims(output, axis=0)  # [1, 1, nk, M1, M2, M3]
  output = jnp.sum(
    output * jnp.abs(_params_w)**2, axis=(3, 4, 5)
  )  # Shape: [2, ni, nk]

  return jnp.sum(output * nocc) / 2
