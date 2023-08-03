mode="direct"
mol="o2"
basis="cc-pvdz"
python3 main.py --run $mode --config.mol_cfg.mol $mol \
 --config.mol_cfg.basis $basis --use_f64 --config.gd_cfg.optimizer rmsprop --config.gd_cfg.meta_opt adam