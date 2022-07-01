# Normalizing Flow for the JDFT in JAX using distrax pacakge
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jssp
import jax.scipy.stats as jss
import distrax
import optax
from jax.nn.initializers import orthogonal
import haiku as hk


def real_nvp(
    input_dim: int,
    ndim: int,
    nlayers: int,
    name: str,
):
  nd = input_dim
  d1 = int(jnp.ceil(nd / 2))
  d2 = nd - d1

  def inner_bijector(params):
    # for volume preserving, the scale needs to be a constant.
    scale, shift = params
    return distrax.ScalarAffine(shift=shift, scale=scale)

  layers = []
  for i in range(nlayers):
    is_swap = lambda i=i: ((i % 2) == 1)
    dim = (d2 if not is_swap() else d1)
    with hk.experimental.name_scope(f"{name}_{i}"):
      mlp = hk.Sequential([
          hk.Linear(ndim),
          jax.nn.gelu,
          hk.Linear(ndim),
          jax.nn.gelu,
          hk.Linear(ndim),
          jax.nn.gelu,
          hk.Linear(2 * dim, w_init=hk.initializers.TruncatedNormal(0.001)),
      ])

    def scale_and_shift(x1, mlp=mlp, dim=dim):
      scale_shift = jnp.reshape(mlp(x1), x1.shape[:-1] + (-1,))
      scale, shift = jnp.split(scale_shift, [dim], axis=-1)
      # scale = jnp.clip(scale, -10., 10.)
      scale = 1.4427 * jax.nn.softplus(scale)
      return scale, shift

    layers.append(
        distrax.SplitCoupling(
            split_index=d1,
            event_ndims=1,
            conditioner=scale_and_shift,
            bijector=inner_bijector,
            swap=is_swap(),
        ))

  return layers


def a_main():
  key = jrandom.PRNGKey(0)
  learning_rate = 0.001
  # optimizer = optax.adam(learning_rate)

  @hk.without_apply_rng
  @hk.transform
  def forward(x):
    layer = distrax.Chain(real_nvp(input_dim=3, ndim=3, nlayers=3, name="nvp"))
    return layer.forward(x)

  @hk.without_apply_rng
  @hk.transform
  def inverse(y):
    layer = distrax.Chain(real_nvp(input_dim=3, ndim=3, nlayers=3, name="nvp"))
    return layer.inverse(y)

  params = forward.init(key, jnp.zeros((1, 3)))
  # opt_state = optimizer.init(params)

  x = jrandom.normal(key, (1, 3))
  y = forward.apply(params, x)
  inverse_y = inverse.apply(params, y)
  print(x, y, inverse_y)


if __name__ == "__main__":
  a_main()
