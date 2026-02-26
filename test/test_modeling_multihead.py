import sys

import torch

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.models import MultiHeadCNNConfig, MultiHeadCNNForClassification

config = MultiHeadCNNConfig(
    head_names={
        "shape": 5,
        "margin": 5,
        "cls": 5,
        "pathology": 2,
    },
    head_dims=[[1024, 512, 256, 128]] * 5,
    gamma={
        "shape": 1.0,
        "margin": 1.5,
        "cls": 1.0,
        "pathology": 2.0,
        "subtlety": 2.0,
    },
    smoothing={
        "shape": 0.05,
        "margin": 0.1,
        "cls": 0.1,
        "pathology": 0.05,
    },
    task_weights={
        "shape": 1.0,
        "margin": 1.2,
        "cls": 1.5,
        "pathology": 2.0,
    },
)
model = MultiHeadCNNForClassification(config)
checkpoint = torch.load("output/experiment-1/best_f1.pth", map_location="cpu")

model.load_state_dict(checkpoint["model_state_dict"])
