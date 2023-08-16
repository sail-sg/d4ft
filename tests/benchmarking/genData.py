
import psi4
import os
import numpy as np

kcalmol = 627.5

path_refdata = '/home/users/nus/sdale/git/refdata/'
path_refdata = '/home/stephen/git/refdata/'

def singleCalc(file_name ='bh76_O'):
    if os.path.isfile(f"{file_name}.out"):
        with open(f"{file_name}.out") as F:
            for f in F:
                if 'Total Energy =' in f:
                    print(f'{file_name:20s} total energy is {float(f.split()[3]):15.6f}')
                    return float(f.split()[3])
    else:
        xyz = "".join(open(path_refdata + "20_bh76/" + file_name + ".xyz", 'r').readlines()[1:]) + 'symmetry c1'
        psi4.set_output_file(f"{file_name}.out")

        psi4.set_options({  'reference' : 'uks',
                            'basis' : '6-31g',
                            'e_convergence' : 1e-8,
                            'soscf' : False,
                            'guess' : 'sad',
                            'maxiter' : 200})

        custom = {"name" : "pbe_custom",
             "x_functionals" : 
                {"GGA_X_PBE" : {}}, 
             "c_functionals" : 
                {"GGA_C_PBE" : {}}
            }

        psi4.geometry(xyz)
        ene = psi4.energy('SCF', dft_functional=custom)
        print(f'{file_name:20s} total energy is {ene:10.4f}')
        return ene

def genEnergy(spec_array):
    energies = []
    for i in spec_array[:,1]:
        energies.append(singleCalc(i))
    result = 0
    for i, c in enumerate(spec_array[:,0]):
        # print(i, c, energies[i])
        result += int(c)*energies[i]
    return result

def dinRead(din_name = path_refdata + '10_din/bh76.din'):
    din_list = []
    with open(din_name) as F:
        for f in F:
            if '#' in f:
                continue
            else:
                spec = []
                spec.append(f[:-1])
                break
        for f in F:
            spec.append(f[:-1])
            for f in F:
                if '0' == f[:-1]:
                    ref = float(next(F)[:-1])
                    break
                else:
                    spec.append(f[:-1])
            spec = np.array(spec).reshape((int(len(spec)/2), 2))
            din_list.append((ref, spec))
            spec = []
    return din_list

def genEnergy(spec_array):
    energies = []
    for i in spec_array[:,1]:
        energies.append(singleCalc(i))
    result = 0
    for i, c in enumerate(spec_array[:,0]):
        # print(i, c, energies[i])
        result += int(c)*energies[i]
    return result

din_list = dinRead()

# error = []
# rxn = din_list[0]
# print(f"calculated energy {genEnergy(rxn[1])*kcalmol:15.6f} kcal/mol")
# print(f" reference energy {rxn[0]:15.6f} kcal/mol")
# error.append(genEnergy(rxn[1])*kcalmol-rxn[0])
# rxn = din_list[1]
# print(f"calculated energy {genEnergy(rxn[1])*kcalmol:15.6f} kcal/mol")
# print(f" reference energy {rxn[0]:15.6f} kcal/mol")
# error.append(genEnergy(rxn[1])*kcalmol-rxn[0])
# rxn = din_list[2]
# print(f"calculated energy {genEnergy(rxn[1])*kcalmol:15.6f} kcal/mol")
# print(f" reference energy {rxn[0]:15.6f} kcal/mol")
# error.append(genEnergy(rxn[1])*kcalmol-rxn[0])
# rxn = din_list[10]
# print(f"calculated energy {genEnergy(rxn[1])*kcalmol:15.6f} kcal/mol")
# print(f" reference energy {rxn[0]:15.6f} kcal/mol")
# error.append(genEnergy(rxn[1])*kcalmol-rxn[0])
# rxn = din_list[26]
# print(f"calculated energy {genEnergy(rxn[1])*kcalmol:15.6f} kcal/mol")
# print(f" reference energy {rxn[0]:15.6f} kcal/mol")
# error.append(genEnergy(rxn[1])*kcalmol-rxn[0])

error = []
for i, d in enumerate(din_list):
    rxn = d
    print(i, d)
    error.append(genEnergy(rxn[1])*kcalmol - rxn[0])
print(error)

import matplotlib.pyplot as plt

plt.boxplot([error, error], whis=(0, 100))
plt.xticks([1, 2], ['psi4', 'd4ft'])
plt.title('pbe 6-31g')
plt.ylabel('Error (kcal/mol)')

plt.savefig('box.png')
plt.close()

from NiceColours import *
from Violins import *

fig, ax = plt.subplots(1)

AddViolin(ax, [1.0, 2.0], [error, error], width=-0.1, dy=1.5,  color_list=["Green", "Orange"])
plt.xticks([1, 2], ['psi4', 'd4ft'])
plt.title('pbe 6-31g')
plt.ylabel('Error (kcal/mol)')

plt.savefig('violin.png')








