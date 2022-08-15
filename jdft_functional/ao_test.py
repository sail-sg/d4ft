"""Test ao.py."""
import jax
import jax.numpy as jnp
import distrax
import numpy as np
from jdft_functional.ao import gaussian_primitive
from absl.testing import absltest, parameterized


class _AOTest(parameterized.TestCase):

  @parameterized.parameters(
    (1., 0, 0, 0), (0.8, 0, 1, 0), (1.5, 1, 0, 0), (2., 2, 0, 0),
    (0.1, 1, 1, 0)
  )
  def test_gaussian_primitive_normalized(self, alpha, i, j, k) -> None:
    key = jax.random.PRNGKey(42)
    temperature = 1.3
    dist = distrax.Normal(0., jnp.sqrt(1. / 2 / alpha / 3) * temperature)
    x = dist.sample(seed=key, sample_shape=(100000, 3))
    sample_prob = jnp.prod(dist.prob(x), axis=-1)
    out = jax.vmap(
      gaussian_primitive, in_axes=(0, None, None)
    )(x, 1., jnp.array([i, j, k]))
    out_square = out**2
    out_weighted = out_square / sample_prob
    np.testing.assert_array_almost_equal(jnp.mean(out_weighted), 1., decimal=2)


if __name__ == "__main__":
  absltest.main()
