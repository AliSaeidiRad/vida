import os
import glob
from pathlib import Path
from warnings import warn
from typing import Dict, Union, Literal

import cv2
import numpy as np
import pandas as pd

from masscls.utils.utils import split_group
from masscls.utils.image import ClockwiseAngleDistance


def get_csaw(
    images: str,
    masks: str,
    screening_data: str,
    output_dir: Union[str, Path],
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    df = {
        "birads": [],
        "shape": [],
        "margin": [],
        "pathology": [],
        "density": [],
        "malignancy": [],
        "JPEGCrop": [],
        "DATASET": [],
    }
    metadata = pd.read_csv(screening_data)

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.mkdir(os.path.join(output_dir, "images"))

    for image_path in glob.glob(os.path.join(images, "*.png")):
        filename = os.path.basename(image_path)
        stem, suffix = os.path.splitext(filename)
        mask_path = os.path.join(masks, f"{stem}_mask{suffix}")

        assert os.path.exists(mask_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        assert mask is not None, mask_path
        assert image is not None, image_path

        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            warn(
                f"Image and Mask does not hav the same dimension (image: `{filename}`, mask: `{os.path.basename(mask_path)}`)"
            )
            continue

        _, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        list_of_pts = []
        for ctr in contours:
            list_of_pts += [pt[0] for pt in ctr]  # type: ignore
        center_pt = np.array(list_of_pts).mean(axis=0)
        clock_ang_dist = ClockwiseAngleDistance(center_pt)
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist)
        ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)

        x, y, w, h = cv2.boundingRect(ctr)
        y_min, y_max = y, y + h
        x_min, x_max = x, x + w

        h, w = mask.shape
        pad_left, pad_bottom, pad_right, pad_top = 100, 100, 100, 100
        x_min = max(0, x_min - pad_left)
        y_min = max(0, y_min - pad_top)
        x_max = min(w, x_max + pad_right)
        y_max = min(h, y_max + pad_bottom)
        roi = image[y_min:y_max, x_min:x_max]

        crop_path = os.path.join(output_dir, "images", os.path.basename(image_path))
        cv2.imwrite(crop_path, roi)

        row = metadata[metadata["anon_filename"] == f"{stem}.dcm"]
        assert len(row) == 1

        x_case = str(row["x_case"].iloc[0])

        if x_case == "0":
            pathology = "BENIGN"
        elif x_case == "1":
            pathology = "MALIGNANT"
        else:
            raise ValueError()

        df["pathology"].append(pathology)
        df["JPEGCrop"].append(crop_path)

        # NOTE: Dummy values to avoid bugs
        # NOTE: for training this dataset, you should freeze learning for these heads in multiheadcnn
        # 20 Feb 2026
        df["birads"].append("0")
        df["density"].append("1")
        df["margin"].append("OBSCURED")
        df["shape"].append("ROUND")
        df["malignancy"].append("LOW")

        # Extra(s) information
        df["DATASET"].append("csaw")

    df = pd.DataFrame(df)

    train, val, test = split_group(df, 0.6, 0.2, 0.2)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(output_dir / f"{name}.csv", index=False)

    return {"train": train, "val": val, "test": test}


__all__ = ["get_csaw"]
