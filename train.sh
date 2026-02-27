#!/usr/bin/env bash

PYTHON="/home/adminteam/Documents/envs/vida/bin/python"

# "$PYTHON" train.py \
# --epochs 700 \
# --datasets cbis cdd-cesm \
# --lr0 1e-5 \
# --lrf 1e-7 \
# --batch-size 32 \
# --num-workers 8 \
# --freeze-backbone

# "$PYTHON" train.py \
# --epochs 700 \
# --datasets cbis cdd-cesm \
# --lr0 1e-5 \
# --lrf 1e-7 \
# --batch-size 32 \
# --num-workers 7 \
# --checkpoint output/experiment-1/best_f1.pth \
# --loss config/loss-optimized.json

"$PYTHON" train.py \
--epochs 700 \
--datasets cbis cdd-cesm \
--lr0 1e-5 \
--lrf 1e-8 \
--batch-size 32 \
--num-workers 7 \
--checkpoint output/experiment-1/best_f1.pth \
--loss config/loss-optimized.json