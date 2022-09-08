"""Sampler for quadrature."""

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import jax.random as jrdm


def batch_sampler(grids, weights, batch_size, seed=1):
  weights = jnp.expand_dims(weights, 1)
  gw = jnp.concatenate((grids, weights), axis=1)
  npoint = grids.shape[0]
  batchsize = min(npoint, batch_size)
  key = jrdm.PRNGKey(seed)
  gw = jrdm.permutation(key, gw)
  g = gw[:, :3]
  w = jnp.squeeze(gw[:, 3])
  
  batch_size = min(npoint, batch_size)
  nbatch = int(npoint / batch_size)

  _g = jnp.split(g[:nbatch*batch_size], nbatch)
  _w = jnp.split((w * npoint / batchsize)[:nbatch*batch_size], nbatch)
  
  return list(zip(_g, _w))

def simple_grid(key, limit, cellsize, n=100, verbose=False):
  """Sample meshgrid for integration."""
  cell_num = int(limit * 2 / cellsize)
  if verbose:
    print('Total number of cells: {}'.format(cell_num**3))
  centers = cellsize * jnp.array(
    jnp.meshgrid(
      *[jnp.arange(cell_num) - int(cell_num / 2) for i in np.arange(3)]
    )
  ).reshape(3, -1).transpose()
  select_idx = jax.random.choice(key, len(centers), shape=[n], replace=False)
  sample = jax.random.uniform(key, (n, 3)) * cellsize + centers[select_idx, :]
  return sample


def poisson_disc_sampling_3d(limit=1, radius=0.1, k=30, n=100):
  """Sample from Poisson disc."""
  boundary = jnp.ones([3, 2])
  boundary = boundary.at[:, 0].set(-1)
  boundary = boundary * limit
  grid_r = radius / 3.0**(1 / 2)
  grid_num_per_axis = int(2 * limit / grid_r) + 1

  # init_grid:
  global grid_index_array
  grid_index_array = -jnp.ones(
    [grid_num_per_axis, grid_num_per_axis, grid_num_per_axis], dtype=jnp.int8
  )

  # store the index of output_array coordinates. -1 means not occupied.
  # Otherwise means the index in the sample array.

  sample_array = []
  activate_array = []

  print(str(grid_num_per_axis**3) + ' bins are generated.')

  def Euclidean_distance(r0, r1):
    return jnp.sum((r0 - r1)**2)**0.5

  def random_point_around(r, k=k):
    """WARNING: This is not uniform around r but we can live with it.

    return:
    shape: (k, 3)
    """
    R = np.random.uniform(radius, 2 * radius, k)
    theta = np.random.uniform(0, 2 * np.pi, k)
    phi = np.random.uniform(0, 2 * np.pi, k)

    delta = jnp.array(
      [
        R * jnp.sin(theta) * jnp.cos(phi), R * jnp.sin(theta) * jnp.sin(phi),
        R * jnp.cos(theta)
      ]
    )

    return r + delta.transpose()

  def in_boundary(r):
    return all(r >= boundary[:, 0]) and all(r <= boundary[:, 1])

  def get_neighborhood(sample_coord):
    """Get neighbor coordinates.

    input: shape:(3)
    """
    grid_index = jnp.array(
      [
        ((sample_coord[0] + limit) / grid_r).astype(int),
        ((sample_coord[1] + limit) / grid_r).astype(int),
        ((sample_coord[2] + limit) / grid_r).astype(int)
      ]
    )
    neigh_index = get_index(grid_index)
    index_in_sample_array = vmap(
      lambda idx: jnp.array(grid_index_array[idx[0], idx[1], idx[2]])
    )(
      neigh_index
    )
    index_in_sample_array = index_in_sample_array[index_in_sample_array > -1]
    output = [sample_array[i] for i in index_in_sample_array]
    return jnp.array(output)

  def get_index(grid_index):
    """Compute neighborhood indices.

    return:
      neighborhood indices. shape: at most [125, 3].
    """
    index_ = np.reshape(
      np.array(np.meshgrid(*[np.arange(5) - 2 for i in range(3)])), (3, -1)
    ).transpose()
    index_ = grid_index + index_
    flag = np.all(index_ >= 0, axis=1)
    index_ = index_[np.all(index_ >= 0, axis=1), :]
    index_ = index_.at[jnp.arange(flag.shape[0])[flag]].get()
    flag = np.all(index_ < grid_num_per_axis, axis=1)
    index_ = index_.at[jnp.arange(flag.shape[0])[flag]].get()
    index_ = index_[np.all(index_ < grid_num_per_axis, axis=1), :]
    return jnp.array(index_)

  def in_neighborhood(new_samples):
    """In neighborhood.

    sample_points: k samples. shape: (k, 3)
    neighbors: shape(n, 3)
    return: the first
    """
    for r0 in new_samples:
      neighbors = get_neighborhood(r0)
      if in_boundary(r0):
        if all(
          vmap(lambda r1: Euclidean_distance(r0, r1))(neighbors) >= radius
        ):
          if grid_index_array[int((r0[0] + limit) / grid_r),
                              int((r0[1] + limit) / grid_r),
                              int((r0[2] + limit) / grid_r)] == -1:
            return r0
    return False

  # initilize
  sample_array.append(jnp.array(np.random.rand(3) * 2 * limit - limit))
  activate_array = sample_array.copy()
  grid_index_array = grid_index_array.at[
    int((activate_array[-1][0] + limit) / grid_r),
    int((activate_array[-1][1] + limit) / grid_r),
    int((activate_array[-1][2] + limit) / grid_r)].set(0)

  while len(activate_array) > 0:
    act_index = int(np.random.rand() * len(activate_array))
    current_point = activate_array[act_index]
    new_samples = random_point_around(current_point)
    in_nei = in_neighborhood(new_samples)

    if in_nei is not False:
      sample_array.append(in_nei)
      activate_array.append(in_nei)
      grid_index_array = grid_index_array.at[
        int((in_nei[0] + limit) / grid_r),
        int((in_nei[1] + limit) / grid_r),
        int((in_nei[2] + limit) / grid_r)].set(len(sample_array) - 1)

    else:
      activate_array.pop(act_index)

    if len(sample_array) >= n:
      print(str(len(sample_array)) + ' points generated in total.')
      return jnp.array(sample_array)

  print(str(len(sample_array)) + ' points generated in total.')
  return jnp.array(sample_array)


if __name__ == '__main__':

  plt.figure()
  plt.subplot(1, 1, 1, aspect=1)

  points = poisson_disc_sampling_3d()
  X = [x for (x, y) in points]
  Y = [y for (x, y) in points]
  plt.scatter(X, Y, s=10)
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.show()
