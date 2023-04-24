import jax

jax.config.update("jax_enable_x64", True)
from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from d4ft.integral.obara_saika.kinetic_integral import kinetic_integral
from absl import logging


class _TestKineticIntegral(absltest.TestCase):

  def setUp(self):
    self.a = (np.array([2, 3, 1]), jnp.array([1., 0.2, 0.3]), jnp.array(1.2))
    self.b = (np.array([2, 3, 1]), jnp.array([1., 0.2, 0.3]), jnp.array(1.2))
    # self.b = (np.array([3, 1, 2]), jnp.array([1.4, 0.1, 0.6]), jnp.array(2.))

  def test_kinetic(self):
    k = kinetic_integral(self.a, self.b, use_horizontal=False)
    logging.info("Kinetic from obara saika is %f", k)

  def test_kinetic_grid(self):

    def gto(r, orb):
      r = r - orb[1]
      xyz = r[0]**orb[0][0] * r[1]**orb[0][1] * r[2]**orb[0][2]
      # N = normalization_constant(orb[0], orb[2])
      return xyz * jnp.exp(-orb[2] * jnp.sum(r * r))

    def a(r):
      return gto(r, self.a)

    def b(r):
      return gto(r, self.b)

    def kinetic1(grid, weight):

      def operator_ket(r):
        return -0.5 * jnp.sum(jnp.diagonal(jax.hessian(b)(r)))

      ok = jax.vmap(operator_ket)(grid)
      bra = jax.vmap(a)(grid)
      return jnp.sum(bra * weight * ok)

    def kinetic2(grid, weight):

      def k(r):
        return 0.5 * jnp.sum(jax.grad(a)(r) * jax.grad(b)(r))

      return jnp.sum(weight * jax.vmap(k)(grid))

    g, w = np.polynomial.legendre.leggauss(30)
    g *= 5.
    w *= 5.
    grids = np.stack(np.meshgrid(g, g, g), axis=-1)
    grids = np.reshape(grids, (-1, 3))
    weights = np.stack(np.meshgrid(w, w, w), axis=-1)
    weights = np.reshape(weights, (-1, 3))
    weights = np.prod(weights, axis=1)
    k1 = kinetic1(grids, weights)
    k2 = kinetic2(grids, weights)
    logging.info(
      "Kinetic from quadrature is %f (second order), %f (first order)", k1, k2
    )


if __name__ == "__main__":
  absltest.main()
