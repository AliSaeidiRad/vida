import sys

import pandas as pd

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.utils import prepare_dataset, plot_dataset_distribution

dfs = prepare_dataset(
    datasets={
        "cbis": {
            "csv": "/home/adminteam/Documents/data/CBIS-DDSM: Breast Cancer Image Dataset/csv",
            "jpeg": "/home/adminteam/Documents/data/CBIS-DDSM: Breast Cancer Image Dataset/jpeg",
            "map": "config/cbis.json",
        },
        "vida": {
            "images": "/home/adminteam/Documents/data/VIDA/images",
            "annotations": "/home/adminteam/Documents/data/only_mass_json/",
            "map": "config/vida.json",
        },
        "csaw": {
            "images": "/home/adminteam/Documents/data/CSAW - MASS/csaw/mass_only_images/",
            "masks": "/home/adminteam/Documents/data/CSAW - MASS/csaw/mass_only_masks/",
            "screening_data": "/home/adminteam/Documents/data/CSAW - MASS/CSAW-CC_breast_cancer_screening_data.csv",
        },
        "cdd-cesm": {
            "annotations": "/home/adminteam/Documents/data/CDD-CESM - MASS/Radiology-manual-annotations.xlsx",
            "segmentations": "/home/adminteam/Documents/data/CDD-CESM - MASS/Radiology_hand_drawn_segmentations_v2.csv",
            "images": "/home/adminteam/Documents/data/CDD-CESM - MASS/images",
            "map": "config/cdd-cesm.json",
        },
    },
    output="temp",
)

# total = pd.concat([_df for _, _df in dfs.items()])

# for dataset in ["cbis", "vida", "csaw", "cdd-cesm"]:
#     for feature in ["shape", "margin", "pathology", "birads"]:
#         print("-" * 100)
#         print(dataset.upper())
#         print(total[total["DATASET"] == dataset][feature].value_counts())
#         print("-" * 100)


# fig = plot_dataset_distribution(dfs, ["shape", "margin", "pathology", "birads"])
# fig.savefig("all.png")
