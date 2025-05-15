#!/usr/bin/env sh
python main.py --config d4ft/config.py:KS-GD-MOL --use_f64 \
    --config.solver_cfg.basis_optim center \
    --config.sys_cfg.basis sto-3g \
    --config.intor_cfg.incore=False \
    --config.solver_cfg.epochs 20000 \
    --config.wandb \
    $@
