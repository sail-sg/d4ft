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
"""energy functions"""
import jax.numpy as jnp
from d4ft.types import PWCoeff
from d4ft.utils import vmap_to_3d


def E_external_pw(
  density_vec_total_ft, atom_coord, atom_charge, gvec, g_mask, vol
):
  """externel energy for plane wave orbitals.
  .. math::
        E = \\sum_G \\vert \\sum_i s_i(G) v_i(G) \\vert n(G)
        S_i(G) = exp(jG\\tau_i)
        v_i(G) = -4 pi q_i / \\Vert G \\Vert^2
  Args:
    density_vec_ft (6d array): the FT of density. Shape: [N1, N2, N3]
    atom_coord (2D array): coordinate of atoms in a unit cell. Shape: [na, 3]
    atom_charge (1D array): charge of atoms in a unit cell. Shape: [na]
    nocc (3D array): occupation mask. Shape: [2, ni, nk]
    gvec (4D array): G-vector. Shape: [N1, N2, N3, 3]
  Return:
    External energy.Float
  """
  g_vec_square = jnp.sum(gvec**2, axis=-1)  # [N1, N2, N3]
  S_i = jnp.exp(
    1j * jnp.matmul(gvec, atom_coord.transpose())
  )  # [N1, N2, N3, na]
  v_i = atom_charge[None, None, None, :] / (g_vec_square[:, :, :, None] + 1e-16)
  v_i *= 4 * jnp.pi  # [N1, N2, N3, na]
  _n_G = density_vec_total_ft * g_mask
  output = _n_G * jnp.abs(jnp.sum(S_i * v_i, axis=-1))  # [N1, N2, N3]
  output = output.at[0, 0, 0].set(0)
  output /= vol
  return -jnp.sum(output).real


def E_hartree_pw(density_vec_total_ft, gvec, g_mask, vol):
  """Hartree energy for plane wave orbitals on reciprocal space.
      E = 2\\pi \\sum_i \\sum_k \\sum_G \\dfrac{|n(G)|^2}{|G|^2}
  Args:
      density_vec_ft (3D array): the FT of density. Shape: [N1, N2, N3]
      gvec (4D array): G-vector. Shape: [N1, N2, N3, 3]
      g_mask (3D array): G point mask, Shape: [N1, N2, N3]
      vol: scalar
  Return:
      Hartree energy: float
  """
  g_vec_square = jnp.sum(gvec**2, axis=-1)  # [N1, N2, N3]
  output = jnp.abs(density_vec_total_ft)**2
  output /= (g_vec_square + 1e-16)
  output *= g_mask
  output = output.at[0, 0, 0].set(0)
  output = jnp.sum(output) / vol / 2 * 4 * jnp.pi
  return output.real


def lda_x_raw(r0: jnp.ndarray):
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.220446049250313e-16, t8 * 2.220446049250313e-16, 1)
  t11 = r0**(0.1e1 / 0.3e1)
  t15 = jnp.where(r0 / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  res = 0.2e1 * 1. * t15
  return res


def E_lda_pw(density_real_vec, vol):
  N1, N2, N3 = density_real_vec.shape
  N = N1 * N2 * N3
  output = jnp.sum(lda_x_raw(density_real_vec) * density_real_vec)
  return output * vol / N


def E_xc_pw(density_real_fun, rvec, vol, polarized=False):
  """xc energy for plane wave orbitals.
  Args:
      density_real_fun (Callable): [3] -> [2] if polarized, else [3] -> [1]
      rvec (4D array): r-vector. Shape: [N1, N2, N3, 3]
      jax_xc_fun (Callabel)
  """
  N1, N2, N3, _ = rvec.shape
  N = N1 * N2 * N3
  exc = lda_x_raw(density_real_fun, polarized=polarized)  # [3]->[1]

  def f(r):
    return jnp.sum(exc(r) * density_real_fun(r))

  return jnp.sum(vmap_to_3d(f)(rvec)) * vol / N * 0.5
  # Not sure where this 0.5 come from. Just saw it from python pw code.


def get_pw_intor(
  pw: PW,
  intor: Literal["obsa", "libcint", "quad"] = "obsa",
  incore_energy_tensors: Optional[ETensorsIncore] = None,
) -> PWIntors:

  def kin_fn(pw_coeff: PWCoeff) -> Float[Array, ""]:
    return e_kin

  def ext_fn(pw_coeff: PWCoeff) -> Float[Array, ""]:
    return e_ext

  def har_fn(pw_coeff: PWCoeff) -> Float[Array, ""]:
    return e_har
