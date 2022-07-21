import numpy as np
import jax
import jax.numpy as jnp
import optax
from jdft.functions import decov
from jdft.orbitals.basis import Basis


class MO_qr(Basis):
  """Molecular orbital using QR decomposition."""

  def __init__(self, nmo, ao, intor=None):
    """Initialize molecular orbital with QR decomposition."""
    super().__init__()
    self.ao = ao
    self.nmo = nmo
    if not intor:
      # self.basis_decov = decov(self.ao.overlap())
      raise AssertionError
    else:
      self.intor = intor
      self.intor.wave_fun = self.ao
      self.basis_decov = decov(self.ao.overlap(intor))
      self.params = self.init(jax.random.PRNGKey(123))

  def init(self, rng_key):
    """Initialize the parameter required by this class."""
    mo_params = jax.random.normal(rng_key,
                                  [self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(rng_key)

  def __call__(self, params, r):
    """Compute the molecular orbital on r.

    R^3 -> R^N. N-body molecular orbital wave functions.
    input: (N: the number of atomic orbitals.)
      |params: N*N
      |r: (3)
    output:
      |molecular orbitals:(2, N)
    """
    mo_params, ao_params = params
    mo_params = jnp.expand_dims(mo_params, 0)
    mo_params = jnp.repeat(mo_params, 2, 0)
    ao_fun_vec = self.ao(r, ao_params)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose() @ decov(
        self.ao.overlap(self.intor)
      ) @ ao_fun_vec  # (self.basis_num)

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)

  def update(self, grad, optimizer, opt_state):
    '''
    Args:
      grad: gradients
      optimizer: optax optimizer object
      opt_state: optax optimizer state objects.
    Returns:
      params: ndarray
      opt_state: if input opt_state is not None, will return updated opt_state.
    '''
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(self.params, updates)
    return params, opt_state
