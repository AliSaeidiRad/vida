import sys
import json

import pandas as pd

df = pd.concat(
    [
        pd.read_csv("data/cbis/train.csv"),
        pd.read_csv("data/cbis/test.csv"),
        pd.read_csv("data/cbis/val.csv"),
    ]
)


def ConvertSubtlety(subtlety):
    if subtlety >= 3:
        return "LOW"
    else:
        return "HIGH"


def ConvertBIRADS(birads):
    if birads in [0, 4, 5]:
        return "MALIGNANT"
    else:
        return "BENIGN"


df["subtlety"] = df["subtlety"].apply(ConvertSubtlety)
df["birads"] = df["birads"].apply(ConvertBIRADS)

print(df[["subtlety", "birads"]].value_counts())
