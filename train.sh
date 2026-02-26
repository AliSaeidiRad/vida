#!/usr/bin/env bash

PYTHON="/home/adminteam/Documents/envs/vida/bin/python"

"$PYTHON" train.py \
--epochs 700 \
--datasets cbis cdd-cesm \
--label2id config/label2id.json \
--lr0 1e-4 \
--lrf 1e-8 \
--batch-size 32 \
--num-workers 7 \
--freeze-backbone
# --freeze-heads pathology \
# --freeze-attention
# --checkpoint output/experiment-?/best_f1.pth \
# --only-prepare-data
# --amp

"$PYTHON" train.py \
--epochs 700 \
--datasets cbis cdd-cesm \
--label2id config/label2id.json \
--lr0 1e-4 \
--lrf 1e-8 \
--batch-size 32 \
--num-workers 7 \
--checkpoint output/experiment-1/best_f1.pth \
# --freeze-heads pathology \
# --freeze-attention
# --only-prepare-data
# --amp