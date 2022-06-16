from ast import Not
import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from pyscf.dft import gen_grid

def set_diag_zero(x):
  return x.at[jnp.diag_indices(x.shape[0])].set(0)

class Intor():
  def __init__(self, mo):
    '''
    Args:
      |mo: molecular orbitals. R^3 -> R^N. N-body molecular orbital wave functions.
      |mol: a molecular object.
    '''
    self.mo = mo

  def single1(self):
    '''
      single particle integral
      \int v(x) \psi dx
    '''
    raise NotImplementedError()

  def single2(self):
    '''
      single particle integral
      \int v(x) \psi^2 dx
    '''
    raise NotImplementedError()

  def double1(self):
    '''
      double particle integral
      \int \psi v(x) \psi dx
    '''
    raise NotImplementedError()

  def double2(self):
    '''
      double particle integral
      \int |\psi|^2 v(x) |\psi|^2 dx
    '''
    raise NotImplementedError()


class LebedevQuadrature(Intor):
  def __init__(self, mo, grids, weights):
    '''
      Args:
        |mo: molecular orbitals. R^3 -> array the output of mo can be any shape. Example: shape=(2, N)
        |mol: a molecular object.
        |level: int. 0~9. grid level.
    '''
    super().__init__(mo)
    self.grids = grids
    self.weights = weights

  def single1(self, v):
    '''
      single particle integral with respect to wave function.
      \int v(x) \psi dx
      Args:
        |v: integrand. R^3 -> R or R^D, where D is shape of the output of mo.
      Returns:
        float. the integral.
    '''

    w_grids = jax.vmap(self.mo)(self.grids)
    v_grids = jax.vmap(v)(self.grids)

    if len(w_grids.shape) > len(v_grids.shape):
      v_grids = jnp.expand_dims(v_grids, axis=np.arange(len(w_grids.shape)-1)+1)

    output = v_grids * w_grids
    output *= jnp.expand_dims(self.weights, axis=np.arange(len(w_grids.shape)-1)+1)
    output = jnp.sum(output)
    return output

  def single2(self, v):
    '''
      single particle integral with respect to pdf.
      \int v(x) \psi^2 dx
      Args:
        |v: integrand. R^3 -> R
      Returns:
        float. the integral.
    '''

    w_grids = jax.vmap(self.mo)(self.grids) ** 2
    v_grids = jax.vmap(v)(self.grids)

    if len(w_grids.shape) > len(v_grids.shape):
      v_grids = jnp.expand_dims(v_grids, axis=np.arange(len(w_grids.shape)-1)+1)

    output = v_grids * w_grids
    output *= jnp.expand_dims(self.weights, axis=np.arange(len(w_grids.shape)-1)+1)
    output = jnp.sum(output)
    return output

  def double2(self, v):
    '''
      double particle integral
      \int \int |n(x) v(x, y) n^2(y) dx dy
      Args:
        |v: integrand. a double variant function. (R^3, R^3) -> R eg: v(x, y) = 1/jnp.norm(x-y)
      Returns:
        float. the integral.
    '''
    outer = lambda x: jnp.outer(x, x)

    w_grids = jax.vmap(self.mo)(self.grids)**2
    w_grids = jnp.sum(w_grids, axis = 1+np.arange(len(w_grids.shape)-1))
    w_mat = outer(w_grids) # shape   R x R

    v_mat = jax.vmap(lambda x: jax.vmap(lambda y: v(x, y))(self.grids))(self.grids)
    v_mat = set_diag_zero(v_mat)    # shape R x R

    weight_mat = outer(self.weights)   # shape R x R

    output = w_mat * v_mat * weight_mat
    output = jnp.sum(output)
    return output

  def lda(self):
    wave_at_grid = jax.vmap(self.mo)(self.grids)  # shape: (D, 2, N)
    const = -3 / 4 * (3 / jnp.pi)**(1 / 3)
    density = jnp.sum(wave_at_grid**2, axis=(1, 2))
    return const * jnp.sum(density**(4 / 3) * self.weights)












