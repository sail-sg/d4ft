import jax

jax.config.update("jax_enable_x64", True)
from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from d4ft.integral.obara_saika.kinetic_integral import kinetic_integral


class _TestKineticIntegral(absltest.TestCase):

  def setUp(self):
    self.a = (np.array([2, 3, 1]), jnp.array([1., 0.2, 0.3]), jnp.array(1.2))
    self.b = (np.array([3, 1, 2]), jnp.array([1.4, 0.1, 0.6]), jnp.array(2.))

  def test_kinetic(self):
    print(kinetic_integral(self.a, self.b))


if __name__ == "__main__":
  absltest.main()
