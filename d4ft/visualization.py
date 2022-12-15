# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code for visualization."""

import sys
import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib

sys.path.append('..')


def save_contour(mol, file, limit=1, delta=0.01):
  """Plot and save the contour."""
  x = np.arange(-limit, limit, delta)
  y = np.arange(-limit, limit, delta)
  z = np.arange(-limit, limit, delta)
  X, Y, Z = np.meshgrid(x, y, z)

  dens_mesh = vmap(
    vmap(
      lambda x, y, z: mol.get_density(
        jnp.
        concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), 1)
      )
    )
  )(X, Y, Z)

  fig, ax = plt.subplots(figsize=(6, 6))
  X, Y = np.meshgrid(x, y)
  # ax.mplot3d(X, Y, Z, jnp.log(dens_mesh))
  ax.text(
    -limit * 0.95, limit * 0.85, 'E: {:.3f}'.format(mol.Egs), c='w', size=16
  )
  im = ax.contourf(X, Y, jnp.log(jnp.mean(dens_mesh, 2)), cmap='magma')
  norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
  sm = plt.cm.ScalarMappable(norm=norm, cmap=im.cmap)

  im.set_clim([-3, 3])
  fig.colorbar(sm)
  fig.savefig(file)
