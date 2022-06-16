'''
Energy functionals.
'''
import jax
from jax import vmap
import jax.numpy as jnp
from jdft.functions import *


def E_kinetic(intor):
  mo = intor.mo
  def f(r):
    hessian_diag = jnp.diagonal(jax.hessian(mo)(r), 0, 2, 3)
    return jnp.sum(hessian_diag, axis=2)
  return -intor.single1(f)/2


def E_ext(intor, nuclei):
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  def v(r):
    return jnp.sum(nuclei_charge/jnp.linalg.norm(r-nuclei_loc, axis=1))
  return -intor.single2(v)


def E_hartree(intor):
  def v(x, y):
    return 1/(jnp.linalg.norm(x-y)+1e-15)
  return intor.double2(v)/2


def E_xc_lda(intor):
  return intor.lda()


def E_nuclear(nuclei):
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  dist_nuc = distmat(nuclei_loc)
  charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
  charge_outer = set_diag_zero(charge_outer)
  return jnp.sum(charge_outer / (dist_nuc + 1e-15)) / 2


def E_gs(intor, nuclei):
  E1 = E_kinetic(intor)
  E2 = E_ext(intor, nuclei)
  E3 = E_xc_lda(intor)
  E4 = E_hartree(intor)
  E5 = E_nuclear(nuclei)
  return E1 + E2 + E3 + E4 + E5, (E1, E2, E3, E4, E5)
