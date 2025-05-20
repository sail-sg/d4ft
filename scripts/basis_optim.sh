#!/usr/bin/env sh
python main.py --config d4ft/config.py:HF-GD-MOL --use_f64 \
    --config.solver_cfg.basis_optim center_flob,coeff,exponent \
    --config.sys_cfg.basis sto-3g \
    --config.intor_cfg.incore=False \
    --config.solver_cfg.epochs 20000 \
    $@
