import numpy as np

from jaxtyping import Array, Float, Int
from typing import List


def get_cart_angular_vec(l: int) -> List[np.ndarray[Int, "3"]]:
  r"""Calculate all l-th order monomials of x,y,z, represented as 3D vectors.
  For example, xyz is represented as [1,1,1].

  Used for the angular part of GTOs.

  Args:
    l: total angular momentum
  """
  vecs = []
  for lx in reversed(range(l + 1)):
    for ly in reversed(range(l + 1 - lx)):
      lz = l - lx - ly
      vecs.append(np.array([lx, ly, lz]))
  return vecs


def racah_norm(l: int):
  r"""Racah's normalization for total angular momentum of :math:`l`,
  denoted as :math:`R(l)`.

  It is used to define normalized spherical harmonics :math:`C_{lm}` where
  :math:`C_{00}=R(0)Y_{00}=1`.
  The formula is
  :math:`R(l)=\sqrt{\frac{4\pi}{2l+1}}`

  The solid harmonic :math:`\mathcal{Y}_{lm}=r^l*Y_{lm}` uses the same
  normalization, and also the real solid harmonics s_{lm}, since they are
  obtained via unitary tranformation from the solid harmonic.
  """
  return np.sqrt((4 * np.pi) / (2 * l + 1))


def real_solid_sph_cart_prefac(l: int, m: int) -> float:
  r"""The prefactor of the real solid harmonics under Cartesian coordinate,
  mulitplied by inverse of Racah's normalization for total angular momentum of
  :math:`l`: :math:`R(l)`.

  GTO basis with variable exponents is defined as

  .. math::
    \chi^{GTO}_{\alpha_{nl}lm}
  = R^{GTO}_{\alpha_{nl}lm}(r)Y_{lm}(\theta, \varphi)

  Absorbing the :math:`r^l` term from the radial part into the spherical
  harmonic :math:`Y_{lm}`, we have the solid harmonic

  .. math::
    C_{lm}(\vb{r})=r^l Y_{lm}(\theta, \varphi)

  This is a complex function. We can get the real solid harmonic :math:`S_{lm}`,
  which is real-valued, via an unitary transformation. The real solid harmonic
  can be expressed in Cartesian coordinate as polynomials of :math:`x,y,z`.
  For example, for the d shell (:math:`l=2`), we have

  .. math::
    S_{22}(\vb{r})=&\frac{1}{2}\sqrt{3}(x^2-y^2) \\
    S_{21}(\vb{r})=&\sqrt{3}xz \\
    S_{20}(\vb{r})=&\frac{1}{2}(2z^2-x^2-y^2) \\
    S_{2-1}(\vb{r})=&\sqrt{3}yz \\
    S_{2-2}(\vb{r})=&\sqrt{3}xy

  By convention Racah's normalization are applied so that we have nice monomial
  for s and p orbitals: for s we have 1 and for p we have :math:`x,y,z`.
  Therefore to convert back to the original spherical harmonics we need to
  undo it. So we need to multiply :math:`1/R(l)`.

  We store the prefactor and the Racah's normalization in the table, which
  are all you need to create real solid harmonics in cartesian coordinates
  with monomials. For example, in the case of :math:`S_{20}`, we have
   :math:`\frac{1}{2} * \frac{1}{R(2)}=0.315391565252520002`.

  Reference:
   - Helgaker 6.6.4
   - https://onlinelibrary.wiley.com/iucr/itc/Bb/ch1o2v0001/table1o2o7o1/

  Args:
    l: total angular momentum
    m: magnetic quantum number
  """
  pass
