import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap


def regularizer(wave_fun, meshgrid, params, N, eps=1e-10):
    f = lambda x: wave_fun(params, x, N)
    wave_at_grid = vmap(f)(meshgrid)    # shape: (D, N)
    col = jnp.matmul(wave_at_grid.transpose(1, 0), wave_at_grid)
    I_N = jnp.eye(N)
    
    return jnp.sum((col - I_N)**2) 
    
    
    
    