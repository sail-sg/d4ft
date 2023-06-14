import einops
import jax.numpy as jnp
from d4ft.utils import complex_norm_square
from jaxtyping import Array, Float, Int


def kinetic_integral(
  reciprocal_lattice_vec: Float[Array, "x y z 3"],
  k_pts: Float[Array, "k 3"],
  pw_coeff: Float[Array, "spin ele k x y z"],
  nocc: Int[Array, "2 ele k"],
) -> Float[Array, ""]:
  """Kinetic energy.

.. math::
      E = 1/2 \sum_{G} |k + G|^2 c_{i,k,G}^* c_{i,k,G}

  Args:
      nocc (3D array): occupation mask. Shape: [2, ni, nk]
  Returns:
      scalar
  """
  # add spin, electron and k dimension
  G = einops.rearrange(reciprocal_lattice_vec, "x y z 3 -> 1 1 1 x y z 3")
  # add spin, electron and lattice dimensions
  k = einops.rearrange(k_pts, "k 3 -> 1 1 k 1 1 1 3")
  kG_norm = einops.reduce(
    (k + G)**2, "spin ele k x y z 3 -> spin ele k x y z", "sum"
  )
  coeff_norm = complex_norm_square(pw_coeff)
  kin_sik = einops.reduce(
    kG_norm * coeff_norm, "spin ele k x y z -> spin ele k", "sum"
  )
  kin = 0.5 * jnp.sum(kin_sik * nocc)
  return kin
