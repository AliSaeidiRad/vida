import sys
import json
import shutil

import pandas as pd

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.utils import prepare_dataset, compute_all_class_weights

all_datasets = json.load(open("config/datasets.json", "r"))

# shutil.rmtree("temp")

dfs = prepare_dataset(
    {
        k: v
        for k, v in all_datasets.items()
        if k
        in [
            "cbis",
            # "cdd-cesm",
            # "csaw",
            # "vida"
        ]
    },
    output="temp",
)

for name, split in dfs.items():
    print(name.upper())
    print(split["shape"].value_counts())
    print(split["margin"].value_counts())
    print(split["birads"].value_counts())
    print(split["pathology"].value_counts())
    print(split["malignancy"].value_counts())
    print(split["subtlety"].value_counts())

# class_weights = compute_all_class_weights(
#     pd.concat([dfs["train"], dfs["val"], dfs["test"]]),
#     json.load(open("config/label2id.json")),
#     smoothing=0.1,
# )

# print(class_weights)

# cbis NOTE: shape, margin, birads, pathology, malignancy
# cdd-cesm NOTE: shape, margin, birads, pathology, malignancy
# vida NOTE: shape, margin, birads, malignancy
# csaw NOTE: pathology
# vindr NOTE: birads, malignancy

# columns = {
#     "cbis": ["shape", "margin", "birads", "pathology", "malignancy"],
#     "cdd-cesm": ["shape", "margin", "birads", "pathology", "malignancy"],
#     "vida": ["shape", "margin", "birads", "malignancy"],
#     "csaw": ["pathology"],
#     "vindr": ["birads", "malignancy"],
# }

# counts = []
# for dataset in all_datasets.keys():
#     for name, split in dfs.items():
#         for column in columns[dataset]:
#             _counts = split[column].value_counts().to_frame()
#             _counts["label"] = column
#             _counts["phase"] = name
#             _counts["dataset"] = dataset
#             counts.append(_counts)
# counts = pd.concat(counts)
# counts.to_csv("counts.csv")
