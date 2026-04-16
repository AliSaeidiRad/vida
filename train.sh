#!/usr/bin/env bash

PYTHON="/home/adminteam/Documents/envs/vida/bin/python"

# STAGE 1

# "$PYTHON" train.py \
# --epochs 1000 \
# --datasets cbis cdd-cesm \
# --label2id config/label2id.json \
# --preprocess iss \
# --batch-size 32 \
# --num-workers 8 \
# --lr0 1e-5 \
# --lrf 1e-7 \
# --freeze-backbone \
# --loss config/loss.json \
# --patience 100 \

# Last Commit STAGE 1: Wed 15 Apr, From 12:09:30 to 13:56:01, Elapsed time 01:46:30

# STAGE 2

# "$PYTHON" train.py \
# --epochs 1000 \
# --datasets cbis cdd-cesm \
# --label2id config/label2id.json \
# --preprocess iss \
# --batch-size 32 \
# --num-workers 8 \
# --lr0 1e-6 \
# --lrf 1e-8 \
# --loss config/loss-optimized.json \
# --patience 100 \
# --checkpoint output/experiment-1/best_f1.pth

# Wed 15 Apr, From 14:50:46 to 21:25:37, Elapsed time 06:34:50

# STAGE 3 - PART I

# "$PYTHON" train.py \
# --epochs 1000 \
# --datasets cbis cdd-cesm vida \
# --label2id config/label2id.json \
# --preprocess iss \
# --batch-size 32 \
# --num-workers 8 \
# --lr0 1e-5 \
# --lrf 1e-8 \
# --loss config/loss-optimized.json \
# --freeze-heads pathology \
# --freeze-backbone \
# --freeze-attention \
# --checkpoint output/experiment-2/best_f1.pth

# Wed 15 Apr, From 21:46:41 to 22:39:17, Elapsed time 00:52:35

# STAGE 3 - PART II

# "$PYTHON" train.py \
# --epochs 1000 \
# --datasets cbis cdd-cesm vida vindr \
# --label2id config/label2id.json \
# --preprocess iss \
# --batch-size 32 \
# --num-workers 8 \
# --lr0 1e-5 \
# --lrf 1e-8 \
# --loss config/loss-optimized.json \
# --freeze-heads pathology shape margin \
# --freeze-backbone \
# --freeze-attention \
# --checkpoint output/experiment-3/best_f1.pth

# Wed 15 Apr, From 22:46:26 to 00:41:32, Elapsed time 01:55:05