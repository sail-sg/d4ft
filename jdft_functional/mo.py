import jax
import jax.numpy as jnp
from jdft.functions import decov
from jdft.orbitals.basis import Basis


class MO_qr(Basis):
  """Molecular orbital using QR decomposition."""

  def __init__(self, nmo, ao):
    """Initialize molecular orbital with QR decomposition."""
    super().__init__()
    self.ao = ao
    self.nmo = nmo

  def init(self, rng_key):
    """Initialize the parameter required by this class."""
    mo_params = jax.random.normal(rng_key,
                                  [self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(rng_key)

  def __call__(self, params, r, **args):
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
          self.ao.overlap(**args)
      ) @ ao_fun_vec  # (self.basis_num)

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)


class MO_vpf(Basis):

  def __init__(self, nmo, ao, vpf):
    self.ao = ao
    self.vpf = vpf
    self.nmo = nmo

  def init(self, rng_key):
    keys = jax.random.split(rng_key, 3)
    mo_params = jax.random.normal(keys[0],
                                  [self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(keys[1]), self.vpf.init(keys[2])

  def __call__(self, params, r, **args):
    """Compute the molecular orbital on r.

    R^3 -> R^N. N-body molecular orbital wave functions.
    input: (N: the number of atomic orbitals.)
      |params: N*N
      |r: (3)
    output:
      |molecular orbitals:(2, N)
    """
    mo_params, ao_params, vpf_params = params
    mo_params = jnp.expand_dims(mo_params, 0)
    mo_params = jnp.repeat(mo_params, 2, 0)

    ao_fun_vec = self.ao(self.vpf(vpf_params, r), ao_params)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose() @ decov(
          self.ao.overlap(**args)
      ) @ ao_fun_vec  # (self.basis_num)

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)
