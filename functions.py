# import jax
from jax import vmap
import jax.numpy as jnp
# from scipy.special import factorial2 as factorial2

def euclidean_distance(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))

def r2(x, y):
    return jnp.sum((x-y)**2)

def distmat(x, y=None):
    '''
    distance matrix
    '''
    if y is None:
        y = x
    return vmap(lambda x1: vmap(lambda y1: euclidean_distance(x1, y1))(y))(x)

    
def gaussian_intergral(alpha, n):
    '''
    ref: https://mathworld.wolfram.com/GaussianIntegral.html
    return \int x^n exp(-alpha x^2) dx 
    
    '''
    
    # if n==0:
    #     return jnp.sqrt(jnp.pi/alpha)
    # elif n==1:
    #     return 0
    # elif n==2:
    #     return 1/2/alpha * jnp.sqrt(jnp.pi/alpha)
    # elif n==3:
    #     return 0
    # elif n==4:
    #     return 3/4/alpha**2 * jnp.sqrt(jnp.pi/alpha)
    # elif n==5:
    #     return 0
    # elif n==6:
    #     return 15/8/alpha**3 * jnp.sqrt(jnp.pi/alpha)
    # elif n==7:
    #     return 0
    # else:
    #     raise NotImplementedError()
    
    return (n==0)*jnp.sqrt(jnp.pi/alpha) + \
        (n==2)*1/2/alpha * jnp.sqrt(jnp.pi/alpha) + \
        (n==4)*3/4/alpha**2 * jnp.sqrt(jnp.pi/alpha) + \
        (n==6)*15/8/alpha**3 * jnp.sqrt(jnp.pi/alpha)
    