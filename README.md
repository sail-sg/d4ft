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
python main.py --run direct --config.mol_cfg.mol O2
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
python main.py --run pyscf --config.mol_cfg.mol O2                             
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
python main.py --run reaction --reaction hf_h_hfhts --config.gd_cfg.gd_cfg.lr_decay cosine
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
dft_cfg: !!python/object:config_config.DFTConfig
  __pydantic_initialised__: true
  rks: false
  rng_seed: 137
  xc_type: lda_x
gd_cfg: !!python/object:config_config.GDConfig
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
mol_cfg: !!python/object:config_config.MoleculeConfig
  __pydantic_initialised__: true
  basis: sto-3g
  charge: 0
  geometry_source: cccdbd
  mol: o2
  spin: -1
scf_cfg: !!python/object:config_config.SCFConfig
  __pydantic_initialised__: true
  epochs: 100
  momentum: 0.5
```

All configuration stated in `d4ft/config.py` can be overridden by providing an appropriate flag (of the form `--config.<cfg_field>`). For example, to change the basis set to `6-31g`, use the flag `--config.mol_cfg.basis 6-31g`. You can directly change the 
`d4ft/config.py` file, or specify a custom config file by supplying the flag `--config <your config file path>`.

## Specifying spin multiplicity
By default all electrons are maximally paired, so the spin is 0 or 1. To specify the spin multiplicity, use the flag `--config.mol_cfg.spin`, for example
``` shell
python main.py --run direct --config.mol_cfg.mol O2 --config.mol_cfg.spin 2
```

## Specifying XC functional
D4FT uses [`jax-xc`](https://github.com/sail-sg/jax_xc) for XC functional. Use the flag `--config.dft_cfg.xc_type` to specify XC functional to use, for example:
``` shell
python main.py --run direct --config.mol_cfg.mol O2 --config.dft_cfg.xc_type lda_x
```


## Specifying Custom Geometries
By default, D4FT uses experimental geometries for molecules from [Computational Chemistry Comparison and Benchmark DataBase](https://cccbdb.nist.gov/). Some examples is stored in `d4ft/system/xyz_files`, for example:
``` python
o2_geometry = """
O 0.0000 0.0000 0.0000;
O 0.0000 0.0000 1.2075;
"""
```
For geometries not cached in the above file, D4FT will query the `cccdbd` website, and you shall see the following logs (using `--config.mol_cfg.mol ch4` in this example):
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
--config.mol_cfg.mol <path_to_geometry_file>
```

# Using the D4FT API directly
If you want to use D4FT inside your program, it is best to call the APIs directly instead of using the `main.py` script. For example, the following is a minimal example of call
direct optimization DFT with D4FT:
``` python
from absl import logging

from d4ft.config import get_config
from d4ft.solver.drivers import incore_cgto_direct_opt_dft

# enable float 64
from jax.config import config
config.update("jax_enable_x64", True)

# make log visible
logging.set_verbosity(logging.INFO) 

# load the default configuration, then override it
cfg = get_config()
cfg.mol_cfg.mol = 'H2'
cfg.mol_cfg.basis = '6-31g'

# Calculation
e_total, _, _ = incore_cgto_direct_opt_dft(cfg)
print(e_total)
```
The `incore_cgto_direct_opt_dft` is just an example of how to use the low level API of `D4FT`, similar to the example models in deep learning libraries. If you want more granular control you should write your function, and you can start by modifying this example.

# Benchmark Against Psi4 and PySCF

We have benchmarked the calculation against well known open-sourced quantum chemsitry libraries: [Psi4](https://psicode.org/) and [PySCF](https://pyscf.org/). 

To run systems from `refdata` benchmark sets, 

``` shell
python main.py --benchmark bh76 --use_f64 --config.mol_cfg.basis <basis> --config.dft_cfg.xc_type <xc> --save --config.mol_cfg.geometry_source refdata --pyscf --config.save_dir <path>
```

To visualize the run:
``` shell
python main.py --run viz --config.save_dir _exp/bh76,6-31g+lda_x,3aahmyt0
```

As shown below, currently D4FT aligns well under the `lda+sto-3g` setting, but it align less well for more complex basis set and / or XC functional. 
D4FT is still undergoing intensive development, so expect these number to improve quite a lot! 

| Reaction          | Basis / XC                    | System     | Psi4 Energy       |                        | Pyscf Energy      |                        | D4FT Energy       |                        | D4FT Setting       |
|-------------------|-------------------------------|------------|-------------------|------------------------|-------------------|------------------------|-------------------|------------------------|--------------------|
|                   |                               |            | Total Energy (Ha) | RXN Energy  (kcal/mol) | Total Energy (Ha) | RXN Energy  (kcal/mol) | Total Energy (Ha) | RXN Energy  (kcal/mol) |                    |
| HF+H->HFH         | LDA_x STO-3G                  | HF         | -97.531437        |                        | -97.530932        |                        | -97.530720        |                        | cosine             |
|                   |                               | H          | -0.411526         |                        | -0.411379         |                        | -0.411379         |                        | cosine             |
|                   |                               | HFH        | -97.934861        |                        | -97.934411        |                        | -97.934166        |                        | cosine             |
|                   |                               | Calculated |                   | 5.084029               |                   | 4.957480               |                   | 4.978091               |                    |
|                   |                               | Reference  |                   | 42.100000              |                   | 42.100000              |                   | 42.100000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE STO-3G | HF         | -98.819667        |                        | -98.819196        |                        | -98.819200        |                        | cosine             |
|                   |                               | H          | -0.464523         |                        | -0.464376         |                        | -0.464376         |                        | cosine             |
|                   |                               | HFH        | -99.274052        |                        | -99.273604        |                        | -99.273606        |                        | cosine             |
|                   |                               | Calculated |                   | 6.361625               |                   | 6.254693               |                   | 6.255669               |                    |
|                   |                               | Reference  |                   | 42.100000              |                   | 42.100000              |                   | 42.100000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE 6-31G  | HF         | -100.302074       |                        | -100.302025       |                        | -100.062894       |                        | cosine             |
|                   |                               | H          | -0.497432         |                        | -0.497432         |                        |                   |                        | cosine             |
|                   |                               | HFH        | -100.765751       |                        | -100.765697       |                        | -100.294316       |                        | cosine             |
|                   |                               | Calculated |                   | 21.181364              |                   | 21.184356              |                   |                        |                    |
|                   |                               | Reference  |                   | 42.100000              |                   | 42.100000              |                   | 42.100000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | LDA_X  6-31G                  | HF         | -99.041733        |                        | -99.041700        |                        | -98.524870        |                        | cosine             |
|                   |                               | H          | -0.454028         |                        | -0.454027         |                        |                   |                        | cosine             |
|                   |                               | HFH        | -99.465905        |                        | -99.465891        |                        | -99.107428        |                        | cosine             |
|                   |                               | Calculated |                   | 18.734730              |                   | 18.722108              |                   |                        |                    |
|                   |                               | Reference  |                   | 42.100000              |                   | 42.100000              |                   | 42.100000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | LDA_X cc-pVDZ                 | HF         | -97.531400        |                        |                   |                        |                   |                        |                    |
|                   |                               | H          | -0.411500         |                        |                   |                        |                   |                        |                    |
|                   |                               | HFH        | -97.934900        |                        |                   |                        |                   |                        |                    |
|                   |                               | Calculated | 5.020024          | 5.084231               |                   |                        |                   |                        |                    |
|                   |                               | Reference  |                   | 42.100000              |                   | 42.100000              |                   | 42.100000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
| N2O + H -> N2OH   | LDA_x STO-3G                  | N2O        | -179.206198       |                        | -179.205101       |                        | -179.205154       |                        |                    |
|                   |                               | H          | -0.411526         |                        | -0.411379         |                        | -0.411379         |                        |                    |
|                   |                               | N2OH       | -179.626057       |                        | -179.624908       |                        | -179.624660       |                        | meta opt           |
|                   |                               | Calculated |                   | -5.228982              |                   | -5.288409              |                   | -5.099372              |                    |
|                   |                               | Reference  |                   | 17.700000              |                   | 17.700000              |                   | 17.700000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE STO-3G | N2O        | -181.958308       |                        | -181.957218       |                        | -181.928330       |                        |                    |
|                   |                               | H          | -0.464523         |                        | -0.464376         |                        | -0.464376         |                        |                    |
|                   |                               | N2OH       | -182.434221       |                        | -182.433079       |                        | -182.412757       |                        | meta opt           |
|                   |                               | Calculated |                   | -7.147259              |                   | -7.207038              |                   | -12.582266             |                    |
|                   |                               | Reference  |                   | 17.700000              |                   | 17.700000              |                   | 17.700000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE 6-31G  | N2O        | -184.404091       |                        | -184.404097       |                        | -184.086108       |                        | meta opt           |
|                   |                               | H          | -0.497432         |                        | -0.497432         |                        |                   |                        |                    |
|                   |                               | N2OH       | -184.891896       |                        | -184.891893       |                        | -183.679929       |                        | meta opt           |
|                   |                               | Calculated |                   | 6.040971               |                   | 6.046698               |                   |                        |                    |
|                   |                               | Reference  |                   | 17.700000              |                   | 17.700000              |                   | 17.700000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | LDA_X  6-31G                  | N2O        | -181.679859       |                        | -181.679841       |                        | -181.405700       |                        | meta opt           |
|                   |                               | H          | -0.454028         |                        | -0.454027         |                        |                   |                        |                    |
|                   |                               | N2OH       | -182.121173       |                        | -182.121144       |                        | -181.369140       |                        | meta opt + rmsprop |
|                   |                               | Calculated |                   | 7.978073               |                   | 7.984961               |                   |                        |                    |
|                   |                               | Reference  |                   | 17.700000              |                   | 17.700000              |                   | 17.700000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | LDA_X cc-pVDZ                 | N2O        | -181.780800       |                        |                   |                        |                   |                        |                    |
|                   |                               | H          | -0.455700         |                        |                   |                        |                   |                        |                    |
|                   |                               | N2OH       | -182.216900       |                        |                   |                        |                   |                        |                    |
|                   |                               | Calculated | 12.299059         | 12.273645              |                   |                        |                   |                        |                    |
|                   |                               | Reference  |                   | 17.700000              |                   | 17.700000              |                   | 17.700000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
| CH3F + F -> FCH3F | LDA_x STO-3G                  | CH3F       | -135.539359       |                        | -135.538595       |                        | -135.534880       |                        | meta opt           |
|                   |                               | F-         | -96.526882        |                        | -96.526450        |                        | -96.526490        |                        |                    |
|                   |                               | FCH3F-     | -232.230600       |                        | -232.229238       |                        | -232.200530       |                        | meta opt           |
|                   |                               | Calculated |                   | -103.135766            |                   | -103.031653            |                   | -87.323317             |                    |
|                   |                               | Reference  |                   | -0.600000              |                   | -0.600000              |                   | -0.600000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE STO-3G | CH3F       | -137.641154       |                        | -137.640409       |                        | -137.637472       |                        | meta opt           |
|                   |                               | F-         | -97.831416        |                        | -97.831044        |                        | -97.831048        |                        |                    |
|                   |                               | FCH3F-     | -235.628004       |                        | -235.626794       |                        | -235.626722       |                        | meta opt           |
|                   |                               | Calculated |                   | -97.535301             |                   | -97.476559             | -99.271803        |                        |                    |
|                   |                               | Reference  |                   | -0.600000              |                   | -0.600000              |                   | -0.600000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE 6-31G  | CH3F       | -139.546241       |                        | -139.546151       |                        | -138.604462       |                        | meta opt           |
|                   |                               | F-         | -99.648544        |                        | -99.648526        |                        | -99.648494        |                        | meta opt           |
|                   |                               | FCH3F-     | -239.251196       |                        | -239.251149       |                        | -238.261548       |                        | meta opt           |
|                   |                               | Calculated |                   | -35.398072             |                   | -35.436043             | -5.391784         |                        |                    |
|                   |                               | Reference  |                   | -0.600000              |                   | -0.600000              |                   | -0.600000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | LDA_X  6-31G                  | CH3F       | -137.471347       |                        | -137.471275       |                        | -137.367613       |                        | meta opt + rmsprop |
|                   |                               | F-         | -98.394822        |                        | -98.394824        |                        | -98.394621        |                        | meta opt           |
|                   |                               | FCH3F-     | -235.930094       |                        | -235.930108       |                        | -235.320157       |                        | meta opt + rmsprop |
|                   |                               | Calculated |                   | -40.113129             |                   | -40.165756             | 277.404618        |                        |                    |
|                   |                               | Reference  |                   | -0.600000              |                   | -0.600000              |                   | -0.600000              |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
| H + C2H4 -> C2H5  | LDA_x STO-3G                  | H          | -0.411526         |                        | -0.411379         |                        | -0.411379         |                        |                    |
|                   |                               | C2H4       | -75.864870        |                        | -75.864238        |                        | -75.853440        |                        |                    |
|                   |                               | C2H5       | -76.277131        |                        | -76.276379        |                        | -76.265090        |                        | meta opt+rmsprop   |
|                   |                               | Calculated |                   | -0.461215              |                   | -0.478267              |                   | -0.169972              |                    |
|                   |                               | Reference  |                   | 2.000000               |                   | 2.000000               |                   | 2.000000               |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE STO-3G | H          | -0.464523         |                        | -0.464376         |                        | -0.464376         |                        |                    |
|                   |                               | C2H4       | -77.508220        |                        | -77.507583        |                        | -77.505231        |                        |                    |
|                   |                               | C2H5       | -77.973545        |                        | -77.972783        |                        | -77.948069        |                        | meta opt+rmsprop   |
|                   |                               | Calculated |                   | -0.503257              |                   | -0.517457              |                   | 13.514625              |                    |
|                   |                               | Reference  |                   | 2.000000               |                   | 2.000000               |                   | 2.000000               |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | 1GGA_X_PBE +1GGA_C_PBE 6-31G  | H          | -0.497432         |                        | -0.497432         |                        |                   |                        |                    |
|                   |                               | C2H4       | -78.452847        |                        | -78.452852        |                        | -78.345467        |                        | meta opt+rmsprop   |
|                   |                               | C2H5       | -78.950867        |                        | -78.950855        |                        | -78.656912        |                        | meta opt+rmsprop   |
|                   |                               | Calculated |                   | -0.368972              |                   | -0.358943              |                   |                        |                    |
|                   |                               | Reference  |                   | 2.000000               |                   | 2.000000               |                   | 2.000000               |                    |
|                   |                               |            |                   |                        |                   |                        |                   |                        |                    |
|                   | LDA_X  6-31G                  | H          | -0.454028         |                        | -0.454027         |                        |                   |                        |                    |
|                   |                               | C2H4       | -76.815174        |                        | -76.815163        |                        | -76.756140        |                        | meta opt+rmsprop   |
|                   |                               | C2H5       | -77.268031        |                        | -77.268001        |                        | -77.076185        |                        | meta opt+rmsprop   |
|                   |                               | Calculated |                   | 0.734806               |                   | 0.746521               |                   |                        |                    |
|                   |                               | Reference  |                   | 2.000000               |                   | 2.000000               |                   | 2.000000               |                    |


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

