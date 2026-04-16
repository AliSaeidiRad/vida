import os
import sys
import json

import torch

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.models import MultiHeadCNNConfig, MultiHeadCNNForClassification
from masscls.utils import load_checkpoint, save_checkpoint

# According to the documentations in 16 April 2026
baseline = "experiment-4"
config_weights = {
    "shape": baseline,
    "margin": baseline,
    "birads": baseline,
    "pathology": "experiment-3",
    "malignancy": baseline,
    "backbone": baseline,
    "attention": baseline,
}

output = os.path.join("output", "experiment-999")
os.makedirs(output, exist_ok=True)

config = json.load(open(os.path.join("output", baseline, "config.json"), "r"))
model_config = MultiHeadCNNConfig.from_dict(config["MODELCONFIG"])
model = MultiHeadCNNForClassification(model_config)
load_checkpoint(os.path.join("output", baseline, "best_f1.pth"), model=model)

new_state_dict = {}
for part, experiment in config_weights.items():
    path = os.path.join("output", experiment, "best_f1.pth")
    state_dict = torch.load(path, map_location="cpu")["model_state_dict"]

    if part == "backbone":
        new_state_dict.update(
            {k: v for k, v in state_dict.items() if k.startswith(f"backbone.")}
        )
        print(f"backbone borrowed from `{path}`")
    elif part == "attention":
        new_state_dict.update(
            {k: v for k, v in state_dict.items() if k.startswith(f"attention.")}
        )
        print(f"attention borrowed from `{path}`")
    else:
        new_state_dict.update(
            {k: v for k, v in state_dict.items() if k.startswith(f"heads.{part}.")}
        )
        print(f"{part} borrowed from `{path}`")

print(len(new_state_dict))

model_dict = model.state_dict()
model_dict.update(new_state_dict)
model.load_state_dict(model_dict)

save_checkpoint(model, os.path.join(output, "best_f1.pth"))

config["OUTPUT_DIR"] = output
config["CHECKPOINT"] = None

json.dump(
    config_weights, open(os.path.join(output, "config_weights.json"), "w"), indent=4
)
json.dump(config, open(os.path.join(output, "config.json"), "w"), indent=4)
