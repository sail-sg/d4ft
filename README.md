# D4FT: Differentiable Density Functional Theory

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

## Quick start: Calculating the ground state energy of Oxygen molecule
D4FT aims to provide atomic APIs similar to modern deep learning library like Jax and Pytorch, so that developers can easily write new algorithm by composing existing APIs.

D4FT also provides examples for standard algorithms, similar to the "train" script in deep learning framework, so that users who just want to run existing algorithms can setup the calculation like any other quantum chemistry library.

Let's calculate the ground state energy of Oxygen molecule with direct minimization DFT:
``` shell
python main.py --run direct --config.mol_cfg.mol_name O2
```

and you should see the following log after the calculation has converged:
``` shell
I0629 10:57:29.393578 140645918545728 logger.py:50] Iter: 395
e_total   -145.984650
e_kin      146.055756
e_ext     -406.366943
e_har      101.187576
e_xc       -14.908516
e_nuc       28.047487
time         0.009397
dtype: float64
I0629 10:57:29.400142 140645918545728 sgd.py:83] Converged: True
```
where each component of the ground state energy is printed.

### Using the configuration system of D4FT
D4FT uses [ml_collections](https://github.com/google/ml_collections) to manage configurations. We have just called the script `main.py` to run the direct minimization, which reads the default configuration file `d4ft/config.py`, and apply overrides to the `mol_name` config via the flag `--config.mol_cfg.mol_name O2`.

The configuration used for the calculation will be printed to the console at the start of the run. For example when you run the calculation for Oxygen above using the default configuration, you should see the following:
``` shell
direct_min_cfg: !!python/object:config_config.DirectMinimizationConfig
  __pydantic_initialised__: true
  converge_threshold: 0.0001
  incore: true
  intor: obsa
  quad_level: 1
  rks: true
  xc_type: lda
mol_cfg: !!python/object:config_config.MoleculeConfig
  __pydantic_initialised__: true
  basis: sto-3g
  mol_name: o2
optim_cfg: !!python/object:config_config.OptimizerConfig
  __pydantic_initialised__: true
  epochs: 2000
  lr: 0.01
  lr_decay: piecewise
  optimizer: adam
  rng_seed: 137
```

All configuration stated in `d4ft/config.py` can be overridden by providing an appropriate flag (of the form `--config.<cfg_field>`). For example, to change the basis set to `6-31g`, use the flag `--config.mol_cfg.basis 6-31g`. You can directly change the 
`d4ft/config.py` file, or specify a custom config file by supplying the flag `--config <your config file path>`.


### Benchmarking against PySCF
Now let's test the accuracy of the calculated ground state energy against well-established open-source QC library PySCF. D4FT provides a thin wrapper around PySCF's API: to run the same calculation above of the Oxygen molecule but with PySCF, run:
``` shell
python main.py --run pyscf --config.mol_cfg.mol_name O2                             
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

## Tutorial and Documentation

### Viewing in the Browser

``` shell
cd docs
pip install -r requirements.txt  # install the tools needs to build the website
make html  # generate a static site from the rst and markdown files
sphinx-serve  # run a server locally, so that it can be viewed in browser
sphinx-autobuild docs docs/_build/html
```

#### Auto-build (optional)

```shell
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

### Editing

The `conf.py` has been setup to support markdown, so we can mix `rst` and `md` files in this project. For example, a `test.md` file is created at `docs/math/test.md`, and it is added to `docs/math/index.rst`.

