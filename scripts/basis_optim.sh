#!/usr/bin/env sh
python -m ipdb main.py --config d4ft/config.py:HF-GD-MOL --use_f64 \
    --config.solver_cfg.basis_optim center_flob \
    --config.sys_cfg.basis 3-21g \
    --config.intor_cfg.incore=False \
    --config.solver_cfg.epochs 20000 \
    $@
