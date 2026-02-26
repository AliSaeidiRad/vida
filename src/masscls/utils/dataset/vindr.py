import os
import ast
import glob
from pathlib import Path
from typing import Dict, Union, Literal

import cv2
import pandas as pd


from masscls.utils.utils import split_group


def get_vindr(
    images: str,
    finding: str,
    output_dir: Union[str, Path],
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    df = {
        # "image": [],
        "birads": [],
        "shape": [],
        "margin": [],
        "pathology": [],
        "density": [],
        "JPEGCrop": [],
        "DATASET": [],
    }

    metadata = pd.read_excel(finding, sheet_name="Sheet1")

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.mkdir(os.path.join(output_dir, "images"))

    for image_path in glob.glob(os.path.join(images, "*")):
        filename = os.path.basename(image_path)
        image_id, _ = os.path.splitext(filename)

        for _, row in metadata[metadata["image_id"] == image_id].iterrows():
            findings = [
                k
                for k in ast.literal_eval(row["finding_categories"])
                if k.lower() == "mass"
            ]
            if findings:
                x_min, y_min, x_max, y_max = (
                    int(row["xmin"]),
                    int(row["ymin"]),
                    int(row["xmax"]),
                    int(row["ymax"]),
                )

                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
                assert image is not None, f"{image_path}"
                h, w, _ = image.shape
                pad_left, pad_bottom, pad_right, pad_top = 100, 100, 100, 100
                x_min = max(0, x_min - pad_left)
                y_min = max(0, y_min - pad_top)
                x_max = min(w, x_max + pad_right)
                y_max = min(h, y_max + pad_bottom)
                roi = image[y_min:y_max, x_min:x_max]
                crop_path = os.path.join(output_dir, "images", filename)
                cv2.imwrite(crop_path, roi)

                df["birads"].append(str(row["finding_birads"])[-1])
                df["JPEGCrop"].append(crop_path)

                # NOTE: Dummy values to avoid bugs
                # NOTE: for training this dataset, you should freeze learning for these heads in multiheadcnn
                # 20 Feb 2026
                df["density"].append("1")
                df["margin"].append("OBSCURED")
                df["shape"].append("ROUND")
                df["pathology"].append("BENIGN")

                # Extra(s) information
                df["DATASET"].append("vindr")
                # df["image"].append(image_id)

    df = pd.DataFrame(df)

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


__all__ = ["get_vindr"]
