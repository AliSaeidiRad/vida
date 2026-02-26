import json
from pathlib import Path
from collections.abc import Mapping
from typing import List, Union, Optional, Callable, Literal, Dict, Any

import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset

from masscls.utils.utils import create_onehot


class DatasetDataFrame(Dataset):
    def __init__(
        self,
        df: Union[
            pd.DataFrame,
            str,
            Path,
            List[pd.DataFrame],
            List[str],
            List[Path],
        ],
        map: Union[Dict, str, Path],
        *,
        transform: Optional[A.Compose] = None,
        preprocess: Union[Literal["normal", "iss"], Callable] = "normal",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(df, str) or isinstance(df, Path):
            self.df = pd.read_csv(df)
        elif isinstance(df, list):
            self.df = pd.concat(
                [pd.read_csv(x) if isinstance(x, (str, Path)) else x for x in df],
                ignore_index=True,
            )
        elif isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise ValueError(f"`df` argument is not valid. ({type(df)})")

        self.target = kwargs.pop("target", "JPEGCrop")
        self.df = self.df.dropna(subset=[self.target])

        self.map: Dict[str, Dict[str, int]] = (
            json.load(open(map, "r")) if isinstance(map, (str, Path)) else map
        )

        if callable(preprocess):
            self.preprocess = preprocess
        elif preprocess == "normal":
            from masscls.utils import ISS

            self.iss = ISS()
            self.preprocess = self.iss_preprocess
        elif preprocess == "iss":
            self.preprocess = self.normal_preprocess
        else:
            raise ValueError("`preprocess argument is not valid`")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.ToTensorV2(),
                ]
            )

        assert isinstance(self.df, pd.DataFrame)
        assert isinstance(self.map, dict)
        assert isinstance(self.transform, A.Compose)

        self.df.reset_index(inplace=True, drop=True)

    def normal_preprocess(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)

        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return image

    def iss_preprocess(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)

        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = self.iss(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[
        str,
        Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]],
    ]:
        row = self.df.iloc[index]
        pixel_values: torch.Tensor = self.transform(
            image=self.preprocess(row[self.target])
        )["image"]

        labels = {
            k: torch.as_tensor(
                create_onehot(
                    self.map[k][str(row[k])],
                    len(self.map[k]),
                ),
                dtype=torch.float32,
            )
            for k in self.map
        }

        return {"pixel_values": pixel_values, **labels}
        # return {"pixel_values": pixel_values, "labels": labels} # for `transformers` library


# for `transformers` library
def default_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not features:
        raise ValueError("features list cannot be empty")

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    batch = {}

    # Handle pixel_values (images)
    pixel_values = []
    for feature in features:
        pixel_value = feature["pixel_values"]

        # Convert numpy array to tensor if needed
        if isinstance(pixel_value, np.ndarray):
            pixel_value = torch.from_numpy(pixel_value)

        pixel_values.append(pixel_value)

    # Stack all pixel values into a single tensor
    batch["pixel_values"] = torch.stack(pixel_values, dim=0)

    # Handle labels (dictionary of tensors)
    if "labels" in features[0] and features[0]["labels"] is not None:
        labels_dict = features[0]["labels"]

        if isinstance(labels_dict, dict):
            # Multi-attribute labels (shape, margin, birads, pathology, etc.)
            batch["labels"] = {}

            for attr_name in labels_dict.keys():
                # Stack all tensors for this attribute
                attr_labels = [feature["labels"][attr_name] for feature in features]
                batch["labels"][attr_name] = torch.stack(attr_labels, dim=0)
        else:
            # Single label tensor
            batch["labels"] = torch.stack(
                [feature["labels"] for feature in features], dim=0
            )

    return batch


__all__ = ["DatasetDataFrame", "default_collator"]
