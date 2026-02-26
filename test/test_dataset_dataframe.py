import sys

import torch

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.data import DatasetDataFrame

dataset = DatasetDataFrame(
    "temp/datasets/cbis-ddsm/train-cbis-ddsm.csv",
    map={
        "shape": {
            "IRREGULAR": 0,
            "ARCHITECTURAL_DISTORTION": 1,
            "OVAL": 2,
            "LOBULATED": 3,
            "ROUND": 4,
        },
        "margin": {
            "SPICULATED": 0,
            "ILL_DEFINED": 1,
            "CIRCUMSCRIBED": 2,
            "OBSCURED": 3,
            "MICROLOBULATED": 4,
        },
        "pathology": {"MALIGNANT": 0, "BENIGN": 1},
        "birads": {"0": 0, "2": 1, "3": 2, "4": 3, "5": 4},
    },
)

for data in dataset:
    pass
