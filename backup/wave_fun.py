import jax
import jax.numpy as jnp
import jax.nn as jnn
# from jax import vmap


def wave_fun(params, coordinate, N):
  '''
    Warning: This function do not satify orthonormality. 
    
    N_single particle wave function: (3) --> (N)
    
    input: 
    
    Coordinate: (3) 3-dimensional vector
    N: number of particals
    
    return:
    
    N wave_fun values at input coordinate: (N)
    
    '''

  x = jnp.repeat(coordinate.reshape(1, 3), N, axis=0)  # (N, 3)
  #     x = jnp.ravel(x)   # (3N)
  x = jnp.expand_dims(x, 2)
  for (w, b) in params:
    '''
        first layer:
        w: (N, 3, M) (here can be made N*M where M << N, which preforms dimension reduction)
        b: (N, N)
        '''
    x = jax.lax.batch_matmul(w.transpose([0, 2, 1]), x)
    x = x + b  # (N, M)
    x = jnn.tanh(x)

  return jnp.squeeze(x)  # (N)


# batched_wave_fun = vmap(wave_fun, in_axes=(None, 0, None))
