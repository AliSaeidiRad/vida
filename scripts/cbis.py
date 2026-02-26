import os
import sys
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.utils.dataset import get_cbis
from masscls.utils.image import ClockwiseAngleDistance
from masscls.utils import get_dataset, ISS, prepare_dataset


label2id: dict = json.load(open("config/label2id.json", "r"))
# label2id.update(
#     {
#         "subtlety": {
#             2: 2,
#             1: 1,
#             0: 0,
#         }
#     }
# )
# id2label = {
#     task: {v: k for k, v in mapping.items()} for task, mapping in label2id.items()
# }
args = json.load(open("config/datasets.json", "r"))
# args["cbis"]["map"] = None
# dfs = get_dataset("cbis", args, "temp/cbis")
dfs = prepare_dataset({k: v for k, v in args.items() if k in ["cbis"]}, "temp")

total = pd.concat([dfs["train"], dfs["test"], dfs["val"]])

print(total["subtlety"].value_counts())

# for idx, row in total.iterrows():
#     new_path = os.path.join(
#         "temp",
#         "subt",
#         str(row["subtlety"]),
#         f"{idx}_{os.path.basename(row['JPEGCrop'])}",
#     )
#     if not os.path.exists(os.path.join("temp", "subt", str(row["subtlety"]))):
#         os.mkdir(os.path.join("temp", "subt", str(row["subtlety"])))
#     shutil.copy(row["JPEGCrop"], new_path)

# fig = plt.figure()
# idx = 0
# for _, row in train[train["subtlety"] == 5].iterrows():
#     if idx == 4:
#         break
#     print(1)
#     image = cv2.imread(row["JPEGCrop"])
#     ax = fig.add_subplot(2, 2, idx + 1)
#     ax.imshow(image)
#     idx += 1
# plt.show()
# row = train.iloc[10]

# print(row["subtlety"])
# print(row["shape"])
# print(row["margin"])

# iss = ISS()
# image = cv2.imread(row["JPEGImage"], cv2.IMREAD_COLOR_RGB)
# mask = cv2.imread(row["JPEGMask"], cv2.IMREAD_GRAYSCALE)

# assert mask is not None
# assert image is not None

# # image = iss(image)

# _, thresh = cv2.threshold(mask, 127, 255, 0)
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# list_of_pts = []
# for ctr in contours:
#     list_of_pts += [pt[0] for pt in ctr]  # type: ignore
# center_pt = np.array(list_of_pts).mean(axis=0)
# clock_ang_dist = ClockwiseAngleDistance(center_pt)
# list_of_pts = sorted(list_of_pts, key=clock_ang_dist)
# ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)

# # image = cv2.drawContours(image, [ctr], -1, (0, 255, 0), 2)

# x, y, w, h = cv2.boundingRect(ctr)
# y_min, y_max = y, y + h
# x_min, x_max = x, x + w

# h, w = mask.shape
# pad_left, pad_bottom, pad_right, pad_top = 100, 100, 100, 100
# x_min = max(0, x_min - pad_left)
# y_min = max(0, y_min - pad_top)
# x_max = min(w, x_max + pad_right)
# y_max = min(h, y_max + pad_bottom)
# roi = image[y_min:y_max, x_min:x_max]

# roi = iss(roi)

# plt.imshow(roi)
# plt.show()
# df = get_cbis(
#     csv=args["csv"],
#     jpeg=args["jpeg"],
#     map=args["map"],
#     output_dir="temp/cbis",
#     mapping=True,
#     do_split=False,
# )
# assert isinstance(df, pd.DataFrame)


# def Shape(value):
#     return label2id["shape"][value]


# def Margin(value):
#     return label2id["margin"][value]


# def Birads(value):
#     return label2id["birads"][str(value)]


# def Pathology(value):
#     return label2id["pathology"][value]


# def Subtlety(value):
#     if value >= 4:
#         return 2
#     elif value == 3:
#         return 1
#     else:
#         return 0


# df["shape"] = df["shape"].apply(Shape)
# df["margin"] = df["margin"].apply(Margin)
# df["birads"] = df["birads"].apply(Birads)
# df["pathology"] = df["pathology"].apply(Pathology)
# df["subtlety"] = df["subtlety"].apply(Subtlety)

# X = []
# y = []
# for _, row in df.iterrows():
#     X.append(row["JPEGCrop"])
#     y.append(
#         [row["shape"], row["margin"], row["birads"], row["pathology"], row["subtlety"]]
#     )

# X = np.asarray(X, dtype=object)
# y = np.asarray(y)

# columns = ["shape", "margin", "birads", "pathology", "subtlety"]

# total_df = pd.DataFrame(y, columns=columns)
# print("\nTOTAL COUNTS")
# print(total_df.apply(pd.Series.value_counts))


# def print_readable_counts(y_split, split_name):
#     print(f"\n================ {split_name} ================")

#     label_names = ["shape", "margin", "birads", "pathology", "subtlety"]

#     for i, label in enumerate(label_names):
#         print(f"\n{label.upper()}")

#         values, counts = np.unique(y_split[:, i], return_counts=True)
#         total = counts.sum()

#         for v, c in sorted(zip(values, counts), key=lambda x: x[0]):
#             label_name = id2label[label][v]
#             print(f"{label_name:<25} : {c:4d} ({c/total:.2%})")


# mskf = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=32)
# for train_index, test_index in mskf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# print(X.shape, X_train.shape, X_test.shape)
# print(y.shape, y_train.shape, y_test.shape)

# y_train_df = pd.DataFrame(y_train, columns=columns)
# y_test_df = pd.DataFrame(y_test, columns=columns)

# print_readable_counts(y_train, "TRAIN")
# print_readable_counts(y_test, "TEST")
# print("\nTRAIN COUNTS")
# print(y_train_df.apply(pd.Series.value_counts))

# print("\nTEST COUNTS")
# print(y_test_df.apply(pd.Series.value_counts))
