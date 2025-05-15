#!/usr/bin/env sh
python main.py --config d4ft/config.py:HF-GD-MOL --use_f64 \
    --config.sys_cfg.mol bh76-bh76_n2 \
    --config.sys_cfg.geometry_source refdata \
    --config.wandb \
    $@
