mode="direct"
mol="o2"
basis="cc-pvdz"
python3 main.py --run $mode --config.sys_cfg.mol $mol \
 --config.sys_cfg.basis $basis --use_f64 --config.solver_cfg.optimizer rmsprop --config.solver_cfg.meta_opt adam