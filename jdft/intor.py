"""Base class of integrator."""

from typing import Callable
import jax
from jdft.orbitals import Basis
import numpy as np
import jax.numpy as jnp
from jdft.functions import set_diag_zero


class Intor():
  """Base class of integrator."""

  def __init__(self, wave_fun: Callable[[jnp.ndarray], jnp.ndarray]):
    """Initialize an Intor.

    Args:
      |wave_fun: molecular orbitals. R^3 -> R^N.
           N-body molecular orbital wave functions.
      |mol: a molecular object.
    """
    self.wave_fun = wave_fun

  def single1(self):
    r"""Single particle integral.

    \int v(x) \psi dx
    """
    raise NotImplementedError()

  def single2(self):
    r"""Single particle integral.

    \int v(x) \psi^2 dx
    """
    raise NotImplementedError()

  def double1(self):
    r"""Double particle integral.

    \int\int \psi(x) v(x, y) \psi(y) dxdy
    """
    raise NotImplementedError()

  def double2(self):
    r"""Double particle integral.

    \int\int \psi^2(x) v(x, y) \psi^2(y) dxdy
    """
    raise NotImplementedError()


class Quadrature(Intor):
  """Quadrature Integrator."""

  def __init__(self, wave_fun, grids, weights):
    """Initialize a Quadrature integrator.

    Args:
      |mo: molecular orbitals. R^3 -> array the output of mo can be any shape.
           Example: shape=(2, N)
      |mol: a molecular object.
      |level: int. 0~9. grid level.
    """
    super().__init__(wave_fun)
    self.grids = grids
    self.weights = weights

  @classmethod
  def from_mo(cls, mo_cls: Basis, nocc: jnp.ndarray, params, grids, weights):

    def wave_fun(x):
      return mo_cls(params, x) * nocc

    return cls(wave_fun, grids, weights)

  def single(self, v=lambda x: 1, exponent=1):
    r"""Single particle integral with respect to wave function.

    \sum_{i} \int v(x) \psi_{i}^{exponent}(x) dx
    Args:
      |v: integrand. R^3 -> R or R^D, where D is shape of the output of mo.
    Returns:
      float. the integral.
    """
    # w_grids: evaluation of single electron wave functions
    w_grids = jax.vmap(self.wave_fun)(self.grids)
    if exponent != 1:
      w_grids = w_grids**exponent
    # v_grids: evaluation of the integrand function v
    v_grids = jax.vmap(v)(self.grids)
    if len(w_grids.shape) > len(v_grids.shape):
      # when v is shared for all orbitals, broadcast v_grids to match w_grids
      v_grids = jnp.expand_dims(
        v_grids, axis=np.arange(len(w_grids.shape) - 1) + 1
      )
    output = v_grids * w_grids
    output *= jnp.expand_dims(
      self.weights, axis=np.arange(len(w_grids.shape) - 1) + 1
    )
    output = jnp.sum(output)
    return output

  def single1(self, v=lambda x: 1):
    r"""Single particle integral with respect to wave function.

    \sum_{i} \int v(x) \psi_{i}(x) dx
    Args:
      |v: integrand. R^3 -> R
    Returns:
      float. the integral.
    """
    return self.single(v, exponent=1)

  def single2(self, v=lambda x: 1):
    r"""Single particle integral with respect to pdf.

    \sum_{i} \int v(x) \psi_{i}^2(x) dx
    Args:
      |v: integrand. R^3 -> R
    Returns:
      float. the integral.
    """
    return self.single(v, exponent=2)

  def double(self, v=lambda x, y: 1, exponent=1):
    r"""Double particle integral with respect to wave function.

    \sum_{i} \int \int \psi^{exponent}(x) v(x, y) \psi^{exponent}(y)
    """

    def outer(x):
      return jnp.outer(x, x)

    w_grids = jax.vmap(self.wave_fun)(self.grids)
    if exponent != 1:
      w_grids = w_grids**exponent

    w_grids = jnp.sum(w_grids, axis=1 + np.arange(len(w_grids.shape) - 1))
    w_grids *= self.weights
    w_mat = outer(w_grids)
    v_mat = jax.vmap(lambda x: jax.vmap(lambda y: v(x, y))(self.grids))(
      self.grids
    )
    v_mat = set_diag_zero(v_mat)
    output = w_mat * v_mat
    output = jnp.sum(output)
    return output

  def double1(self, v=lambda x, y: 1):
    r"""Double particle integral: inner product.

    \int \int \psi(x)^T v(x, y) \psi(y) dx dy
    Args:
      |v: integrand. a double variant function. (R^3, R^3) -> R
          eg: v(x, y) = 1/jnp.norm(x-y)
    Returns:
      float. the integral.
    """
    return self.double(v, exponent=1)

  def double2(self, v=lambda x, y: 1):
    """Double2 integrates with square of wave function."""
    return self.double(v, exponent=2)

  def overlap(self):
    """Overlap matrix of the orbitals."""

    w_grids = jax.vmap(self.wave_fun)(self.grids)
    w_grids = jnp.reshape(w_grids, newshape=(w_grids.shape[0], -1))
    w_grids_weighted = w_grids * jnp.expand_dims(self.weights, axis=(1))

    return jnp.matmul(w_grids_weighted.T, w_grids)

  def lda(self):
    """Integrate to compute the LDA functional."""
    wave_at_grid = jax.vmap(self.wave_fun)(self.grids)  # shape: (D, 2, N)
    const = -3 / 4 * (3 / jnp.pi)**(1 / 3)
    density = jnp.sum(wave_at_grid**2, axis=(1, 2))
    return const * jnp.sum(density**(4 / 3) * self.weights)
