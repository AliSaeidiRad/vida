import os
import json
import glob
from pathlib import Path
from warnings import warn
from typing import Dict, Union, Literal

import cv2
import pandas as pd

from masscls.utils.utils import map_columns_values, split_group


def get_vida(
    images: str,
    annotations: str,
    map: Dict[str, Union[str, Dict[Literal["action"], str]]],
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

    for image_path in glob.glob(os.path.join(images, "*/*.png")):
        metadata_path = os.path.join(
            annotations,
            os.path.basename(image_path).replace(".png", ".json"),
        )

        assert os.path.exists(
            metadata_path
        ), f"Corresponded JSON Metadata for image does not exists (image: {image_path})"

        metadata = json.load(open(metadata_path, "r"))

        pad_left, pad_bottom, pad_right, pad_top = 100, 100, 100, 100

        if not os.path.exists(os.path.join(output_dir, "images")):
            os.mkdir(os.path.join(output_dir, "images"))

        for lesion in metadata["shapes"]:
            if lesion["label"].lower() == "mass":
                points = lesion["points"]

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                x_min = int(min(xs))
                y_min = int(min(ys))
                x_max = int(max(xs))
                y_max = int(max(ys))

                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
                if image is None:
                    raise ValueError("image is empty!")
                h, w, _ = image.shape
                x_min = max(0, x_min - pad_left)
                y_min = max(0, y_min - pad_top)
                x_max = min(w, x_max + pad_right)
                y_max = min(h, y_max + pad_bottom)
                roi = image[y_min:y_max, x_min:x_max]
                crop_path = os.path.join(
                    output_dir,
                    "images",
                    os.path.basename(image_path),
                )
                cv2.imwrite(crop_path, roi)

                try:
                    attributes = lesion["attributes"].keys()
                except KeyError:
                    warn(f"Mass Attributes not found in `{metadata_path}`!")
                    continue

                attributes = {k.lower(): v for k, v in lesion["attributes"].items()}
                birads = attributes.get("bi-rads") or attributes.get("bi_rads")
                shape = attributes.get("shape")
                margin = attributes.get("margin")
                density = attributes.get("density")

                df["birads"].append(
                    birads.upper() if isinstance(birads, str) else birads
                )
                df["shape"].append(shape.upper() if isinstance(shape, str) else shape)
                df["margin"].append(
                    margin.upper() if isinstance(margin, str) else margin
                )
                df["pathology"].append("BENIGN")
                df["density"].append(
                    density.upper() if isinstance(density, str) else density
                )
                df["JPEGCrop"].append(crop_path)

                # Extra(s) information
                df["DATASET"].append("vida")
                # df["image"].append(image_path)

    df = pd.DataFrame(df)

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


__all__ = ["get_vida"]
