from enum import unique
from operator import index
from xmlrpc.client import boolean
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt

from math import floor, ceil
import jax.random as jrdm

# from pyscf.dft import radi
# from pyscf.dft import gen_grid


def batch_sampler(grids, weights, batchsize, seed=1):
  '''
    inputs:
    |grids: (N, 3)
    |weights: (N, )
    |factor: percentage that split the grids.
    output:
    a list of ceil(1/factor) with each element is a subset of grids.
    '''

  npoint = grids.shape[0]
  key = jrdm.PRNGKey(seed)
  idx = jrdm.permutation(key, jnp.arange(npoint), independent=True)
  batchsize = min(npoint, batchsize)
  nbatch = int(npoint / batchsize)

  sample_boundary = jnp.arange(nbatch + 1) * batchsize
  sample_boundary = jnp.asarray(sample_boundary, jnp.int32)
  output_grid = []
  output_weight = []
  batch_idx = [
    idx[sample_boundary[i]:sample_boundary[i + 1]] for i in jnp.arange(nbatch)
  ]

  for i in jnp.arange(nbatch):
    output_grid.append(grids[batch_idx[i]])
    output_weight.append(weights[batch_idx[i]] * npoint / batchsize)

  return output_grid, output_weight


# def batch_sampler(grids, weights, factor=0.1, seed=1):

#     '''
#     inputs:
#     |grids: (N, 3)
#     |weights: (N, )
#     |factor: percentage that split the grids.
#     output:
#     a list of ceil(1/factor) with each element is a subset of grids.
#     '''
#     npoint = grids.shape[0]
#     key=jrdm.PRNGKey(seed)

#     nbatch = min(npoint, floor(1/factor))
#     batch_size = floor(npoint/nbatch)
#     # sample_boundary = (jnp.arange(nbatch-1)+1)*batch_size
#     # sample_boundary = jnp.asarray(sample_boundary, jnp.int16)
#     sample_boundary = [int((i+1)*batch_size) for i in range(nbatch)]

#     weights = jnp.expand_dims(weights, axis=1)
#     gw = jnp.concatenate((grids, weights), axis=1)
#     gw = jrdm.permutation(key, gw, axis=0, independent=False)
#     output_grid = []
#     output_weight = []

#     output_grid = jnp.split(gw[:, :3], sample_boundary, axis=0)
#     output_weight = jnp.split(gw[:, 3], sample_boundary, axis=0)
#     output_grid.pop()
#     output_weight.pop()

#     return output_grid, output_weight


def simple_grid(key, limit, cellsize, n=100, verbose=False):

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
  '''
    input:

    '''

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
    # WARNING: This is not uniform around r but we can live with it
    '''
        return:
        shape: (k, 3)
        '''
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
    '''
        get neighbor coordinates
        input: shape:(3)
        return:
        '''

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
    #         index_in_sample_array = jnp.where(index_in_sample_array>-1, index_in_sample_array, 0)
    index_in_sample_array = index_in_sample_array[index_in_sample_array > -1]

    #         print(index_in_sample_array)
    #         index_in_sample_array
    output = [sample_array[i] for i in index_in_sample_array]

    #         output = sample_array[index_in_sample_array]
    return jnp.array(output)

  def get_index(grid_index):
    '''
        compute neighborhood indices
        return:
        neighborhood indices. shape: at most [125, 3].
        '''
    # def axis_gen(i):
    #     if grid_index[i] == 0:
    #         x_ = jnp.arange(3)
    #     elif grid_index[i] == 1:
    #         x_ = jnp.arange(4)-1
    #     elif grid_index[i] == grid_num_per_axis-1:
    #         x_ = jnp.arange(3) - 2
    #     elif grid_index[i] == grid_num_per_axis-2:
    #         x_ = jnp.arange(4) - 2
    #     else:
    #         x_ = jnp.arange(5)-2
    #     return x_

    # grid_index_np = np.asarray(grid_index)
    index_ = np.array(np.meshgrid(*[np.arange(5) - 2 for i in range(3)])
                     ).reshape(3, -1).transpose()
    index_ = grid_index + index_
    flag = np.all(index_ >= 0, axis=1)
    index_ = index_[np.all(index_ >= 0, axis=1), :]
    index_ = index_.at[jnp.arange(flag.shape[0])[flag]].get()
    flag = np.all(index_ < grid_num_per_axis, axis=1)
    index_ = index_.at[jnp.arange(flag.shape[0])[flag]].get()
    index_ = index_[np.all(index_ < grid_num_per_axis, axis=1), :]
    #         grid_index_expand = jnp.expand_dims(grid_index, 0).repeat(index_.shape[0], axis=0)
    #         flag = jnp.expand_dims(jnp.all(index_>=0, axis=1), 1).repeat(3, axis=1)
    #         index_ = jnp.where(flag, index_, grid_index_expand)
    #         flag = jnp.expand_dims(jnp.all(index_<grid_num_per_axis, axis=1), 1).repeat(3, axis=1)
    #         index_ = jnp.where(flag, index_, grid_index_expand)
    return jnp.array(index_)

  def in_neighborhood(new_samples):
    '''
        input:
        sample_points: k samples. shape: (k, 3)
        neighbors: shape(n, 3)
        return: the first
        '''

    #         def f_(r0):
    #             with jax.disable_jit():
    #                 neighbors = get_neighborhood(r0)

    #             if in_boundary(r0):
    #                 # print(vmap(lambda r1: Euclidean_distance(r0, r1))(neighbors))
    #                 if all(vmap(lambda r1: Euclidean_distance(r0, r1))(neighbors) >= radius):
    #                     if grid_index_array[((r0[0]+limit)/grid_r).astype(int),
    #                     ((r0[1]+limit)/grid_r).astype(int),
    #                     ((r0[2]+limit)/grid_r).astype(int)]==-1:
    #                         return True
    #                     else:
    #                         return False
    #                 else:
    #                     return False

    #             else:
    #                 return False

    #         # print(new_samples.shape)
    #         flag = [f_(i) for i in new_samples]
    # #         with jax.disable_jit():
    # #             flag = vmap(f_)(new_samples)
    #         output = new_samples[flag, :]
    #         if output.shape[0]>0:
    #             return output[0]
    #         else:
    #             return False

    for r0 in new_samples:
      neighbors = get_neighborhood(r0)
      if in_boundary(r0):
        # print(vmap(lambda r1: Euclidean_distance(r0, r1))(neighbors))
        # print(r0)
        if all(
          vmap(lambda r1: Euclidean_distance(r0, r1))(neighbors) >= radius
        ):
          if grid_index_array[int((r0[0] + limit) / grid_r),
                              int((r0[1] + limit) / grid_r),
                              int((r0[2] + limit) / grid_r)] == -1:
            return r0
    return False

  ### initilize
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
