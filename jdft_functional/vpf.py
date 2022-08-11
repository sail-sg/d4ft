import jax
import jax.numpy as jnp


class CouplingLayer():
  '''
  Coupling transformation f: x -> y
  y0 = x0
  y1 = x1 + tanh(w10 * x0)
  y2 = x2 + tanh(w20 * x0 + w21 * w1)
  '''

  def __init__(self, activation=True):
    self.mask = jnp.tril(jnp.ones([3, 3]), -1)
    self.activation = activation
    # self.params = jnp.empty([3, 3])

  def __call__(self, params, x):
    # self.params = params
    shift = jnp.matmul(params * self.mask, x)

    if self.activation:
      shift = jnp.tanh(shift)

    y = x + shift
    return y

  def init(self, rng_key):
    return jax.random.normal(rng_key, ([3, 3])) / 1000

  def inverse(self, params, y):
    if self.activation:
      shift_back = jnp.array(
        [
          0,
          jnp.tanh(params[1, 0] * y[0]),
          jnp.tanh(
            params[2, 0] * y[0] + params[2, 1] * (y[1] - params[1, 0] * y[0])
          )
        ]
      )
    else:
      shift_back = jnp.array(
        [
          0, params[1, 0] * y[0],
          params[2, 0] * y[0] + params[2, 1] * (y[1] - params[1, 0] * y[0])
        ]
      )
    x = y - shift_back
    return x


class OrthogonalLayer():
  '''
  Orthogonal transformation
  '''

  def __init__(self):
    self.params = jnp.empty([3, 3])

  def __call__(self, params, x):
    qmat, r = jnp.linalg.qr(params)
    return jnp.matmul(qmat, x)

  def init(self, rng_key):
    # return jax.random.normal(rng_key, ([3, 3])) / 10
    return jnp.eye(3)

  def inverse(self, params, y):
    qmat, r = jnp.linalg.qr(params)
    return jnp.matmul(qmat.T, y)


class VolumePreservingFlow():
  '''
  Volume Proserving flow.
  '''

  def __init__(self, layers='oc' * 3):

    self.layers = []
    for s in layers[:-1]:
      if s == 'c':
        self.layers.append(CouplingLayer())
      elif s == 'o':
        self.layers.append(OrthogonalLayer())
    # deal with the last layer
    if len(layers) > 0:
      if layers[-1] == 'c':
        self.layers.append(CouplingLayer(activation=False))
      elif layers[-1] == 'o':
        self.layers.append(OrthogonalLayer())
    self.depth = len(self.layers)

  def __call__(self, params, x):
    for i in zip(params, self.layers):
      param, layer = i
      x = layer(param, x)
    return x

  def inverse(self, params, x):
    for i in zip(params[::-1], self.layers[::-1]):
      param, layer = i
      x = layer.inverse(param, x)
    return x

  def init(self, rng_key):
    keys = jax.random.split(rng_key, self.depth)
    return [self.layers[i].init(keys[i]) for i in range(self.depth)]
