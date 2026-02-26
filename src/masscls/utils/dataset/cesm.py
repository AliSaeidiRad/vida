import os
import json
from pathlib import Path
from typing import Dict, Union, Literal

import cv2
import numpy as np
import pandas as pd


from masscls.utils.utils import map_columns_values, split_group


def get_cesm(
    annotations: str,
    segmentations: str,
    images: str,
    map: Dict[str, Union[str, Dict[Literal["action"], str]]],
    output_dir: Union[str, Path],
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    def roi_to_bbox(roi_options):
        roi_type = roi_options.get("name")

        if roi_type == "polygon":
            xs = np.array(roi_options["all_points_x"], dtype=np.int32)
            ys = np.array(roi_options["all_points_y"], dtype=np.int32)

            if len(xs) != len(ys):
                raise ValueError("Polygon x and y coordinate lists must match")

            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

        elif roi_type == "ellipse":
            cx = roi_options["cx"]
            cy = roi_options["cy"]
            rx = roi_options["rx"]
            ry = roi_options["ry"]

            x_min = int(cx - rx)
            x_max = int(cx + rx)
            y_min = int(cy - ry)
            y_max = int(cy + ry)

        elif roi_type == "circle":
            cx = roi_options["cx"]
            cy = roi_options["cy"]
            r = roi_options["r"]

            x_min = int(cx - r)
            x_max = int(cx + r)
            y_min = int(cy - r)
            y_max = int(cy + r)

        else:
            raise ValueError(f"Unsupported ROI type: {roi_type}")

        return x_min, y_min, x_max, y_max

    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    annotations_df = pd.concat(
        [
            pd.read_excel(
                annotations,
                sheet_name="mass_description",
            ),
            pd.read_excel(
                annotations,
                sheet_name="mass enhancement_description",
            ),
        ]
    )
    segmentations_df = pd.read_csv(segmentations)

    segmentations_df = segmentations_df[segmentations_df["region_count"] == 1]

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.mkdir(os.path.join(output_dir, "images"))

    df = {
        # "image": [],
        "shape": [],
        "margin": [],
        "pathology": [],
        "birads": [],
        "JPEGCrop": [],
        # "XMIN": [],
        # "YMIN": [],
        # "XMAX": [],
        # "YMAX": [],
        "DATASET": [],
    }

    for idx, row in segmentations_df.iterrows():
        finding = annotations_df[
            annotations_df["Image_name"] == row["#filename"].split(".")[0]
        ]
        if len(finding) == 1:
            namesplit = row["#filename"].split("_")
            image_path = os.path.join(
                images, str(namesplit[0]), f"{namesplit[1]}_{namesplit[-1]}"
            )

            if not os.path.exists(image_path):
                continue

            x_min, y_min, x_max, y_max = roi_to_bbox(
                json.loads(row["region_shape_attributes"])
            )  # image[y_min:y_max, x_min:x_max]

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
            if image is None:
                raise ValueError("image is empty!")
            h, w, _ = image.shape
            pad_left, pad_bottom, pad_right, pad_top = 100, 100, 100, 100
            x_min = max(0, x_min - pad_left)
            y_min = max(0, y_min - pad_top)
            x_max = min(w, x_max + pad_right)
            y_max = min(h, y_max + pad_bottom)
            roi = image[y_min:y_max, x_min:x_max]
            crop_path = os.path.join(
                output_dir, "images", f"{namesplit[0]}_{os.path.basename(image_path)}"
            )
            cv2.imwrite(crop_path, roi)

            df["JPEGCrop"].append(crop_path)
            df["shape"].append(finding["Mass shape"].values[0].upper())
            df["margin"].append(finding["Mass margin"].values[0].upper())
            df["pathology"].append(
                finding["Pathology Classification/ Follow up"].values[0].upper()
            )
            df["birads"].append(finding["BIRADS"].values[0])

            # Extra(s) information
            df["DATASET"].append("cdd-cesm")
            # df["image"].append(image_path)
            # df["XMIN"].append(x_min)
            # df["YMIN"].append(y_min)
            # df["XMAX"].append(x_max)
            # df["YMAX"].append(y_max)

    df = pd.DataFrame(df)
    df["birads"] = df["birads"].astype(str)

    df = map_columns_values(df, map)

    # Extra(s) Head
    def Malignancy(b):
        if str(b) in ["0", "4", "5"]:
            return "HIGH"
        else:
            return "LOW"

    df["malignancy"] = df["birads"].apply(Malignancy)

    train, val, test = split_group(df, 0.6, 0.2, 0.2)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(output_dir / f"{name}.csv", index=False)

    return {"train": train, "val": val, "test": test}


__all__ = ["get_cesm"]
