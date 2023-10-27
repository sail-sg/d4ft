# D4FT: Differentiable Density Functional Theory

Joint work with [i-fim](https://ifim.nus.edu.sg/).

## Installation

1. Create a `virtualenv` with `python3.9`: 
``` shell
virtualenv --python="/usr/bin/python3.9" .venv
```
2. Enter the `virtualenv`: `source .venv/bin/activate`
3. Install `D4FT`
``` shell
pip install -e .
```
4. [Optional] To use GPU, install the GPU version of jax:
``` shell
./scripts/gpu_setup.sh
```

# Quick start: using D4FT as a standalone program
D4FT aims to provide atomic APIs similar to modern deep learning library like Jax and Pytorch, so that developers can easily write new algorithm by composing existing APIs.

D4FT also provides examples for standard algorithms, similar to the "train" script in deep learning framework, so that users who just want to run existing algorithms can setup the calculation like any other quantum chemistry library.

## Calculating the ground state energy of Oxygen molecule
Let's calculate the ground state energy of Oxygen molecule with direct minimization DFT:
``` shell
python main.py --run direct --config.sys_cfg.mol O2
```

and you should see the following log after the calculation has converged:
``` shell
I0728 23:26:17.428046 140634023188288 sgd.py:141] e_total std: 5.535387390409596e-05
I0728 23:26:17.428330 140634023188288 sgd.py:151] Converged: True
I0728 23:26:17.478279 140634023188288 drivers.py:160] lowest total energy: 
 e_total    -146.04742
e_kin       146.03822
e_ext      -406.35187
e_har         101.126
e_xc       -14.907249
e_nuc       28.047487
time         0.010077
Name: 778, dtype: object
```
where each component of the ground state energy is printed.

## Benchmarking against PySCF
Now let's test the accuracy of the calculated ground state energy against well-established open-source QC library PySCF. D4FT provides a thin wrapper around PySCF's API: to run the same calculation above of the Oxygen molecule but with PySCF, run:
``` shell
python main.py --run pyscf --config.sys_cfg.mol O2                             
```
This will call PySCF to perform SCF calculation with the setting stated in `d4ft/config.py`. Two sets of energy will be printed: 
1. the energy calculated with PySCF's integral engine `libcint`, which uses Rys quadrature:
``` shell
**** SCF Summaries ****
Total Energy =                        -145.993023703410159
Nuclear Repulsion Energy =              28.047487783751553
One-electron Energy =                 -260.344533053220744
Two-electron Coulomb Energy =          101.213596609086963
DFT Exchange-Correlation Energy =      -14.909575043027928
```
2. the energy calculated with D4FT's integral engine (where we load the MO coefficients from the PySCF's calculation), which implements the Obara-Saika scheme:
``` shell
e_total   -145.993042
e_kin      146.081512
e_ext     -406.426056
e_har      101.213593
e_xc       -14.909574
e_nuc       28.047487
time         0.000974
dtype: float64
I0630 11:00:55.155952 139685586032448 main.py:120] 1e energy:-260.34454345703125
```
where `1e energy` is the sum of kinetic and external potential energy. We see that the energy value agrees up to 5 decimal places, and that direct minimization finds a wavefunction with lower energy.

## Calculate energy barrier for reaction

``` shell
python main.py --run reaction --reaction hf_h_hfhts --config.solver_cfg.solver_cfg.lr_decay cosine
```
This calculate the ground state energy for each system, then compute the energy barrier:
``` shell
I0713 14:34:15.694393 140558110832448 main.py:65] e_hf = -97.53069305419922 Ha
I0713 14:34:15.694551 140558110832448 main.py:65] e_h = -0.4113791286945343 Ha
I0713 14:34:15.694614 140558110832448 main.py:65] e_hfhts = -97.93412780761719 Ha
I0713 14:34:15.728916 140558110832448 main.py:66] e_barrier = 0.00794219970703125 Ha = 4.9837541580200195 kcal/mol
```

Another example:
``` shell
python main.py --run reaction --reaction n2o_h_n2ohts
```
which should return
``` shell
I0713 15:30:42.618415 139621268744000 main.py:65] e_n2o = -179.2051544189453 Ha
I0713 15:30:42.618541 139621268744000 main.py:65] e_h = -0.4113791286945343 Ha
I0713 15:30:42.618597 139621268744000 main.py:65] e_n2ohts = -179.59417724609375 Ha
I0713 15:30:42.644445 139621268744000 main.py:66] e_barrier = 0.0223541259765625 Ha = 14.027280807495117 kcal/mol
```

# Using the configuration system of D4FT
D4FT uses [ml_collections](https://github.com/google/ml_collections) to manage configurations. We have just called the script `main.py` to run the direct minimization, which reads the default configuration file `d4ft/config.py`, and apply overrides to the `mol` config via the flag `--config.mol_cfg.mol O2`.

The configuration used for the calculation will be printed to the console at the start of the run. For example when you run the calculation for Oxygen above using the default configuration, you should see the following:
``` shell
method_cfg: !!python/object:config_config.MethodConfig
  __pydantic_initialised__: true
  restricted: false
  rng_seed: 137
  xc_type: lda_x
solver_cfg: !!python/object:config_config.GDConfig
  __pydantic_initialised__: true
  converge_threshold: 0.0001
  epochs: 4000
  hist_len: 50
  lr: 0.01
  lr_decay: piecewise
  meta_lr: 0.03
  meta_opt: none
  optimizer: adam
intor_cfg: !!python/object:config_config.IntorConfig
  __pydantic_initialised__: true
  incore: true
  intor: obsa
  quad_level: 1
sys_cfg: !!python/object:config_config.MoleculeConfig
  __pydantic_initialised__: true
  basis: sto-3g
  charge: 0
  geometry_source: cccdbd
  mol: o2
  spin: -1
solver_cfg: !!python/object:config_config.SCFConfig
  __pydantic_initialised__: true
  epochs: 100
  momentum: 0.5
```

All configuration stated in `d4ft/config.py` can be overridden by providing an appropriate flag (of the form `--config.<cfg_field>`). For example, to change the basis set to `6-31g`, use the flag `--config.sys_cfg.basis 6-31g`. You can directly change the 
`d4ft/config.py` file, or specify a custom config file by supplying the flag `--config <your config file path>`.

## Specifying spin multiplicity
By default all electrons are maximally paired, so the spin is 0 or 1. To specify the spin multiplicity, use the flag `--config.sys_cfg.spin`, for example
``` shell
python main.py --config.sys_cfg.mol O2 --config.sys_cfg.spin 2
```

## Specifying XC functional
D4FT uses [`jax-xc`](https://github.com/sail-sg/jax_xc) for XC functional. Use the flag `--config.method_cfg.xc_type` to specify XC functional to use, for example:
``` shell
python main.py --config.sys_cfg.mol O2 --config.method_cfg.xc_type lda_x
```


## Specifying Custom Geometries
By default, D4FT uses experimental geometries for molecules from [Computational Chemistry Comparison and Benchmark DataBase](https://cccbdb.nist.gov/). Some examples is stored in `d4ft/system/xyz_files`, for example:
``` python
o2_geometry = """
O 0.0000 0.0000 0.0000;
O 0.0000 0.0000 1.2075;
"""
```
For geometries not cached in the above file, D4FT will query the `cccdbd` website, and you shall see the following logs (using `--config.sys_cfg.mol ch4` in this example):
``` shell
I0630 11:12:49.016396 140705043318592 cccdbd.py:108] **** Posting formula
I0630 11:12:50.397949 140705043318592 cccdbd.py:116] **** Fetching data
I0630 11:12:51.625333 140705043318592 mol.py:47] spin: 0, geometry: C  0.0000 0.0000 0.0000
H  0.6276 0.6276 0.6276
H  0.6276 -0.6276 -0.6276
H  -0.6276 0.6276 -0.6276
H  -0.6276 -0.6276 0.6276
```
To use custom geometries, first create a plain text file with name `<mol_name>.xyz`, for example `h2.xyz`
``` text
H 0.0000 0.0000 0.0000;
H 0.0000 0.0000 0.7414;
```
then pass it through the config flag as follows

``` shell
--config.sys_cfg.mol <path_to_geometry_file>
```

## Switching algorithms
To load config for other algorithms, do
``` shell
--config d4ft/config.py:<config-string>
```
Some examples:
1. Kohn-Sham DFT with SCF
``` shell
python main.py --run scf --config d4ft/config.py:KS-SCF-MOL --config.sys_cfg.mol O --config.method_cfg.xc_type "1*gga_c_pbe+1*gga_x_pbe" --use_f64 --config.solver_cfg.momentum 0.5
```
2. Direct optimization Hartree-Fock
``` shell
python main.py --config d4ft/config.py:HF-GD-MOL --use_f64 --pyscf --config.sys_cfg.mol bh76-bh76_n2 --config.sys_cfg.geometry_source refdata
```

# Using the D4FT API directly
If you want to use D4FT inside your program, it is best to call the APIs directly instead of using the `main.py` script. For example, the following is a minimal example of call
direct optimization DFT with D4FT:
``` python
from absl import logging

from d4ft.config import get_config
from d4ft.solver.drivers import cgto_direct

# enable float 64
from jax.config import config
config.update("jax_enable_x64", True)

# make log visible
logging.set_verbosity(logging.INFO) 

# load the default configuration, then override it
cfg = get_config()
cfg.sys_cfg.mol = 'H2'
cfg.sys_cfg.basis = '6-31g'

# Calculation
e_total, _, _ = cgto_direct(cfg)
print(e_total)
```
The `cgto_direct` is just an example of how to use the low level API of `D4FT`, similar to the example models in deep learning libraries. If you want more granular control you should write your function, and you can start by modifying this example.

# Benchmark Against Psi4 and PySCF

We have benchmarked the calculation against well known open-sourced quantum chemsitry libraries: [Psi4](https://psicode.org/) and [PySCF](https://pyscf.org/). 

To run systems from `refdata` benchmark sets, 

``` shell
python main.py --benchmark bh76 --use_f64 --config.sys_cfg.basis <basis> --config.method_cfg.xc_type <xc> --save --config.sys_cfg.geometry_source refdata --pyscf --config.save_dir <path>
```

To visualize the run:
``` shell
python main.py --run viz --config.save_dir _exp/bh76,6-31g+lda_x,3aahmyt0
```

## Tutorial and Documentation

### Viewing in the Browser

``` shell
cd docs
pip install -r requirements.txt  # install the tools needs to build the website
make html  # generate a static site from the rst and markdown files
sphinx-serve  # run a server locally, so that it can be viewed in browser
```

#### Auto-build (optional)

At the root directory, run the following
```shell
pip install sphinx-autobuild
sphinx-autobuild --watch d4ft docs docs/_build/html
```

### Editing

The `conf.py` has been setup to support markdown, so we can mix `rst` and `md` files in this project. For example, a `test.md` file is created at `docs/math/test.md`, and it is added to `docs/math/index.rst`.

