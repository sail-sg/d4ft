import jax
from jax import vmap
import jax.numpy as jnp
from jdft.functions import *


def E_kinetic(wave_fun, meshgrid, params, weight):

    '''
    E_gs = T_s + V_ext + V_Hartree + V_xc
    This function is to compute the kinetic energy T_s.

    input:
        meshgrid: (D, 3)
        wave_fun: (3)-->(2, N) wave function.
        params: parameter array for wavefun
        N: number of electrons.
        weight: (D)
    output:
        kinetic_energy: scalar
    '''

    f = lambda x: wave_fun(params, x)
    def laplacian_3d(r):

        '''
        this function computes the laplacian operator

        1/2 \partial^2 f/\partial x^2 + \partial^2 f/\partial y^2 + \partial^2 f/\partial y^2

        which is the diagonal of heissen matrix
        # TODO: This can be more efficient without calculating the off-diagonal entries.
        input
            r: shape (3)

        output: scalar
        '''

        # hessian_diag = jnp.diagonal(jax.jacfwd(jax.jacrev(f))(r), 0, 2, 3)
        # hessian shape: (2, N, 3, 3)
        hessian_diag = jnp.diagonal(jax.hessian(f)(r), 0, 2, 3)
        return jnp.sum(hessian_diag, axis=2)/2

    wave_at_grid = vmap(f)(meshgrid)   # shape: (D, 2, N)
    batched_lap = vmap(laplacian_3d)(meshgrid)    # shape: (D, 2, N)
    return -jnp.sum(batched_lap * wave_at_grid * jnp.expand_dims(weight, [1, 2]))


def E_kinetic_2(wave_fun, meshgrid, params, weight):

    '''
    E_gs = T_s + V_ext + V_Hartree + V_xc
    This function is to compute the kinetic energy T_s.

    input:
        meshgrid: (D, 3)
        wave_fun: (3)-->(2, N)  a function that calculate the wave function.
        params: parameter array for wavefun
        N: number of electrons.
        weight: (D)
    output:
        kinetic_energy: scalar
    '''

    f = lambda x: wave_fun(params, x)
    def laplacian_3d(r):

        '''
        this function computes the laplacian operator

        1/2 \partial^2 f/\partial x^2 + \partial^2 f/\partial y^2 + \partial^2 f/\partial y^2

        which is the diagonal of heissen matrix
        # TODO: This can be more efficient without calculating the off-diagonal entries.
        input
            r: shape (3)

        output: scalar
        '''

        # hessian_diag = jnp.diagonal(jax.jacfwd(jax.jacrev(f))(r), 0, 2, 3)
        # hessian shape: (2, N, 3)
        hessian_diag = jax.jacrev(f)(r)**2
        return jnp.sum(hessian_diag, axis=2)/2

    wave_at_grid = vmap(f)(meshgrid)   # shape: (D, 2, N)
    batched_lap = vmap(laplacian_3d)(meshgrid)    # shape: (D, 2, N)
    return -jnp.sum(batched_lap * wave_at_grid * jnp.expand_dims(weight, [1, 2]))


def E_ext(wave_fun, meshgrid, nuclei, params, weight, eps=1e-10):
    '''
    input:
        wave_fun: (3)-->(2, N)
        meshgrid: (D, 3)
        nuclei: dict {'loc': jnp.array [A, 3], 'charge':jnp.array or a list of scalar}
        weight: (D)
    output:
        scalar
    '''

    f = lambda x: wave_fun(params, x)
    wave_at_grid = vmap(f)(meshgrid) # shape: (D, 2, N)
    nuclei_loc = nuclei['loc']
    nuclei_charge = nuclei['charge']
    num_ncl = len(nuclei_loc)
    output = 0

    for k in range(num_ncl):
        x = jnp.abs(meshgrid-nuclei_loc[k].reshape((1, 3)))   # shape: (D, 3)
        x = jnp.sum(x**2, axis=1)**0.5          #shape: (D)
        x = - nuclei_charge[k]/(x + eps)
        output += jnp.sum(wave_at_grid**2 * jnp.expand_dims(x, [1, 2]) * jnp.expand_dims(weight, [1, 2]))

    return output   # shape: (2, N)


def E_XC_LDA(wave_fun, meshgrid, params, weight):
    '''
    input:
        meshgrid: (D, 3)
        wave_fun: (3)-->(N)  a function that calculate the wave function.
        params: params for wavefun

    output:
        kinetic_energy: scalar
    '''
    f = lambda x: wave_fun(params, x)
    wave_at_grid = vmap(f)(meshgrid)    # shape: (D, 2, N)
    # density = wave_at_grid**2        # shape: (D, 2, N)
    const = -3/4*(3/jnp.pi)**(1/3)
    # const = -1
    # return jnp.sum((2*density[:, 0, :]) ** (4/3) * jnp.expand_dims(weight, 1)) * const/2 +\
    #     jnp.sum((2*density[:, 1, :]) ** (4/3) * jnp.expand_dims(weight, 1)) * const/2

    density = jnp.sum(wave_at_grid**2, axis=(1, 2))
    return const * jnp.sum(density ** (4/3)*weight)


def E_Hartree(wave_fun, meshgrid, params, weight, eps=1e-10):
    '''
    Warning: this function is computing in a fully pairwised manner, which is prune to out-of-memory issue.
    meshgrid: (D, 3)
    '''

    f = lambda x: wave_fun(params, x)
    # density_at_grid = vmap(f)(meshgrid) # shape: (D,  2,  N)
    # density_at_grid = density_at_grid**2  # shape: (D, 2,  N)
    # density_at_grid *= jnp.expand_dims(weight, [1, 2])

    # density_at_grid = density_at_grid.transpose([2, 1, 0])   # shape: (N, 2, D)

    # dist_pair = distmat(meshgrid)    # shape: (D, D)
    # density_at_grid = jnp.sum(density_at_grid, axis=[0, 1])  # shape: (D)

    # def f(i):
    #     density_pair = jax.lax.batch_matmul(jnp.expand_dims(density_at_grid, 1),
    #                                         jnp.expand_dims(density_at_grid, 0))
    #     return jnp.sum(set_diag_zero(density_pair/(dist_pair+eps)))

    density_at_grid = vmap(f)(meshgrid) # shape: (D,  2,  N)
    density_at_grid = jnp.sum(density_at_grid**2, axis=(1, 2)) * weight
    dist_pair = distmat(meshgrid)

    density_pair = jnp.dot(jnp.expand_dims(density_at_grid, 1),
                                        jnp.expand_dims(density_at_grid, 0))
    output = jnp.sum(set_diag_zero(density_pair/(dist_pair+eps)))
    return output/2

def E_nuclear(nuclei, eps=1e-10):
    nuclei_loc = nuclei['loc']
    nuclei_charge = nuclei['charge']

    dist_nuc = distmat(nuclei_loc)
    charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
    charge_outer = set_diag_zero(charge_outer)

    return jnp.sum(charge_outer/(dist_nuc+eps))/2

def E_gs(wave_fun, meshgrid, params, nuclei, weight, eps=1e-10):

    E1 = E_kinetic(wave_fun, meshgrid, params, weight)
    E2 = E_ext(wave_fun, meshgrid, nuclei, params, weight, eps)
    E3  = E_XC_LDA(wave_fun, meshgrid, params, weight)
    E4 = E_Hartree(wave_fun, meshgrid, params, weight, eps)

    return E1 + E2 + E3 + E4

def set_diag_zero(x):
    return x.at[jnp.diag_indices(x.shape[0])].set(0)

