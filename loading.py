import pyscf
import jax
import jax.numpy as jnp

'''
this module align pyscf results and cdft parameters.
WARNING: only 3-21G is implemented.
'''



def wave_from_pyscf(param, r, pyscf_mol):
    '''
    input: 
        param: N
        (3)-dimensional coordinate
    output: (N) wave function value vector.
    '''

    atom = pyscf_mol
    geometry = atom.atom
    # 'C -0.5, 0.0, 0.0;\nC 0.5, 0., 0.' 
    basis = atom.basis
    basis_param = atom._basis
    # atom_coords = atom._basis
    
    '''
    the basis param format (Pople-type):
    a list of list
    [
        [l, [zeta_11, c_11], [zeta_12, c_12], ....]   ===> psi_s1 = c_11 exp(-zeta_11 x) + ...
        [l, [zeta_21, c_21], [zeta_22, c_22], ....]
        ...
    ]
    
    l: angular momentum quantum number where 0 represent s, 1 presents p.
    TODO:  pre-calculate these constants.
    
    '''
    output = []
    for element in atom.elements:
        for i in atom._basis[element]:
            if i[0] == 0:
                prm_array = jnp.array(i[1:])
                output.append(jnp.sum(prm_array[:, 1]*\
                    jnp.exp(-prm_array[:, 0]*jnp.linalg.norm(r)**2)*\
                        (2*prm_array[:, 0]/jnp.pi)**(3/4)))
                
            elif i[0] == 1:
                prm_array = jnp.array(i[1:])
                output += [r_i *jnp.sum(prm_array[:, 1]*jnp.exp(-prm_array[:, 0]*\
                    jnp.linalg.norm(r)**2) * (2*prm_array[:, 0]/jnp.pi)**(3/4) * (4*prm_array[:, 0])**0.5)\
                    for r_i in r]
    
    # print(jnp.array(output).shape)
    return jnp.dot(param, jnp.array(output))

    
    