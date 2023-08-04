import numpy as np
from pyscf import dft, gto

mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvtz', cart=True)
grids = dft.gen_grid.Grids(mol).build()
ao = mol.eval_gto('GTOval', grids.coords)
print(ao)
s = np.einsum('pi,p,pj->ij', ao, grids.weights, ao)
print(abs(s - mol.intor('int1e_ovlp')).max())
print(np.allclose(s, mol.intor('int1e_ovlp'), atol=1e-7))
