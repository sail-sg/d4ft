# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Solve DFT with gradient descent"""

from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
from absl import logging

import wandb
from d4ft.config import GDConfig
from d4ft.logger import RunLogger
from d4ft.optimize import get_optimizer
from d4ft.types import Hamiltonian, TrainingState, Trajectory, Transition


def get_unique_centers_with_indices(centers, tol=1e-8):
  """Extract unique 3D coordinates and their indices from centers array.

    Args:
        centers: Array of shape (..., 3) containing 3D coordinates
        tol: Tolerance for considering coordinates as identical

    Returns:
        unique_centers: Array of shape (n_unique, 3) containing unique coordinates
        indices: Array containing indices mapping each original coordinate to unique ones
    """
  # Reshape to 2D array (n_points, 3)
  flat_centers = centers.reshape(-1, 3)

  # Round to tolerance to avoid floating point issues
  rounded = jnp.round(flat_centers / tol) * tol

  # Get unique rows and indices
  unique_centers, indices = jnp.unique(rounded, axis=0, return_inverse=True)

  return unique_centers, indices


def scipy_opt(
  solver_cfg: GDConfig, H: Hamiltonian, params: hk.Params, key: jax.Array
) -> float:
  energy_fn_jit = jax.jit(lambda mo_coeff: H.energy_fn(mo_coeff, key)[0])
  import jaxopt
  solver = jaxopt.BFGS(fun=energy_fn_jit, maxiter=500)
  res = solver.run(params)
  return res


def sgd(
  solver_cfg: GDConfig, H: Hamiltonian, params: hk.Params, key: jax.Array
) -> Tuple[RunLogger, Trajectory]:

  @jax.jit
  def update(state: TrainingState) -> Tuple:
    """update parameter, and accumulate gradients"""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    val_and_grads_fn = jax.value_and_grad(H.energy_fn, has_aux=True)
    (loss, aux), grad = val_and_grads_fn(state.params, rng_key)
    energies, mo_grads = aux
    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(
      params, opt_state, next_rng_key
    ), energies, mo_grads

  @jax.jit
  def meta_loss(meta_params: hk.Params, state: TrainingState):
    opt_state = state.opt_state
    opt_state.hyperparams["learning_rate"] = jax.nn.sigmoid(meta_params)
    state = TrainingState(state.params, opt_state, state.rng_key)

    # for _ in range(10):
    _, new_state, energies, mo_grads = update(state)
    # state = new_state

    loss = H.energy_fn(new_state.params, new_state.rng_key)[0]

    return loss, (new_state, energies, mo_grads)

  @jax.jit
  def meta_step(state: TrainingState, meta_state: TrainingState):
    grad, aux = jax.grad(meta_loss, has_aux=True)(meta_state.params, state)
    new_state, energies, mo_grads = aux
    meta_updates, meta_opt_state = meta_opt.update(grad, meta_state.opt_state)
    new_meta_params = optax.apply_updates(meta_state.params, meta_updates)
    return TrainingState(
      new_meta_params, meta_opt_state, meta_state.rng_key
    ), new_state, energies, mo_grads

  # init state
  opt_states = get_optimizer(solver_cfg, params, key)
  optimizer, state = opt_states["main"]
  if solver_cfg.meta_opt != "none":
    meta_opt, meta_state = opt_states["meta"]

  # GD loop
  traj = []
  converged = False
  logger = RunLogger()
  e_total_std = 0.

  # get original atom coords
  if 'center_flob' in state.params['~']:
    centers = state.params['~']['center_flob']
    unique_coords, atom_indices = get_unique_centers_with_indices(centers)

  for step in range(solver_cfg.epochs):

    if solver_cfg.meta_opt == "none":
      loss, new_state, energies, mo_grads = update(state)
      logging.info(f"{loss=}")
    else:
      meta_state, new_state, energies, mo_grads = meta_step(state, meta_state)
      logging.info(f"cur lr: {jax.nn.sigmoid(meta_state.params):.4f}")

    logger.log_step(energies, step, e_total_std)
    logger.get_segment_summary()

    if wandb.run is not None and 'center_flob' in new_state.params['~']:
      cur_centers = state.params['~']['center_flob']
      plot_centers_3d(unique_coords, cur_centers, step)

    mo_coeff = H.mo_coeff_fn(state.params, state.rng_key, apply_spin_mask=False)
    t = Transition(mo_coeff, energies, mo_grads)

    traj.append(t)

    state = new_state

    if step < solver_cfg.hist_len:  # don't check for convergence
      continue

    # check convergence
    e_total_std = jnp.stack(
      [t.energies.e_total for t in traj[-solver_cfg.hist_len:]]
    ).std()
    if e_total_std < solver_cfg.converge_threshold:
      converged = True
      break

  logging.info(f"Converged: {converged}")

  return logger, traj


def plot_centers_3d(unique_coords, cur_centers, step):
  """Create a 3D scatter plot of original atoms and basis centers using plotly.

    Args:
        unique_coords: Array of shape (n_atoms, 3) containing atomic positions
        cur_centers: Array of shape (n_basis, 3) containing basis centers
        step: Current optimization step
    """
  # Convert to numpy for plotting
  atoms = np.array(unique_coords)
  basis = np.array(cur_centers)

  # Create the 3D scatter plot
  fig = go.Figure()

  # Add atoms
  fig.add_trace(
    go.Scatter3d(
      x=atoms[:, 0],
      y=atoms[:, 1],
      z=atoms[:, 2],
      mode='markers',
      name='Atoms',
      marker=dict(size=10, color='blue', symbol='circle')
    )
  )

  # Add basis centers
  fig.add_trace(
    go.Scatter3d(
      x=basis[:, 0],
      y=basis[:, 1],
      z=basis[:, 2],
      mode='markers',
      name='Basis Centers',
      marker=dict(size=6, color='red', symbol='circle')
    )
  )

  # Update layout
  fig.update_layout(
    title=f'Atom and Basis Centers (Step {step})',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    width=800,
    height=800,
    showlegend=True
  )

  # Calculate distances for statistics
  distances = np.linalg.norm(basis[:, None] - atoms[None], axis=2).min(axis=1)

  # Log to wandb
  wandb.log(
    {
      "centers_plot": wandb.Html(fig.to_html()),
      "basis_stats":
        {
          "max_distance_to_atom": float(distances.max()),
          "mean_distance_to_atom": float(distances.mean()),
          "std_distance_to_atom": float(distances.std())
        },
      "step": step
    }
  )
