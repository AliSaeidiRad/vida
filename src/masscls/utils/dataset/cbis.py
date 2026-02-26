import os
import glob
from pathlib import Path
from typing import Dict, Union, Literal

import cv2
import numpy as np
import pandas as pd


from masscls.utils.image import ClockwiseAngleDistance
from masscls.utils.utils import split_group, map_columns_values


def get_cbis(
    csv: str,
    jpeg: str,
    map: Dict[str, Union[str, Dict[Literal["action"], str]]],
    output_dir: Union[str, Path],
    mapping: bool = True,
    do_split: bool = True,
) -> Union[pd.DataFrame, Dict[Literal["train", "val", "test"], pd.DataFrame]]:
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.mkdir(os.path.join(output_dir, "images"))

    dicom_info = pd.read_csv(os.path.join(csv, "dicom_info.csv"))

    df = pd.concat(
        [
            pd.read_csv(os.path.join(csv, "mass_case_description_train_set.csv")),
            pd.read_csv(os.path.join(csv, "mass_case_description_test_set.csv")),
        ]
    )

    # SOPInstanceUID --
    # SeriesInstanceUID -2
    # StudyInstanceUID -3

    def StudyInstanceUID(path):
        return path.split("/")[-3]

    def SeriesInstanceUID(path):
        return path.split("/")[-2]

    def Name(path):
        return path["image file path"].split("/")[0] + "_" + str(path["abnormality id"])

    def JPEGMask(df):
        mask_study = df["MaskStudyInstanceUID"]
        mask_series = df["MaskSeriesInstanceUID"]

        patien_name = df["PatientName"]

        output = dicom_info[dicom_info["PatientName"] == patien_name]
        output = output[output["StudyInstanceUID"] == mask_study]
        output = output[output["SeriesInstanceUID"] == mask_series]
        output = output[output["SeriesDescription"] == "ROI mask images"]

        assert len(output) == 1
        return os.path.join(jpeg, *output.iloc[0]["image_path"].split("/")[2:])

    # def JPEGCrop(df):
    #     mask_study = df["MaskStudyInstanceUID"]
    #     mask_series = df["MaskSeriesInstanceUID"]

    #     patien_name = df["PatientName"]

    #     output = dicom_info[dicom_info["PatientName"] == patien_name]
    #     output = output[output["StudyInstanceUID"] == mask_study]
    #     output = output[output["SeriesInstanceUID"] == mask_series]
    #     output = output[output["SeriesDescription"] == "cropped images"]
    #     if len(output) == 1:
    #         return os.path.join(jpeg, *output.iloc[0]["image_path"].split("/")[2:])
    #     elif len(output) == 0:
    #         return None
    #     else:
    #         raise ValueError

    def JPEGImage(series_uid):
        image_path = list(glob.glob(os.path.join(jpeg, str(series_uid), "*.jpg")))
        assert len(image_path) == 1
        return str(image_path[0])

    def JPEGCCrop(df):
        pname = df["PatientName"]
        image_path = df["JPEGImage"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(df["JPEGMask"], cv2.IMREAD_GRAYSCALE)

        assert mask is not None
        assert image is not None

        _, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        list_of_pts = []
        for ctr in contours:
            list_of_pts += [pt[0] for pt in ctr]  # type: ignore
        center_pt = np.array(list_of_pts).mean(axis=0)
        clock_ang_dist = ClockwiseAngleDistance(center_pt)
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist)
        ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)

        # DEBUG
        # image = cv2.drawContours(image, [ctr], -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(ctr)
        y_min, y_max = y, y + h
        x_min, x_max = x, x + w

        h, w = mask.shape
        pad_left, pad_bottom, pad_right, pad_top = 200, 200, 200, 200
        x_min = max(0, x_min - pad_left)
        y_min = max(0, y_min - pad_top)
        x_max = min(w, x_max + pad_right)
        y_max = min(h, y_max + pad_bottom)
        roi = image[y_min:y_max, x_min:x_max]

        crop_path = os.path.join(
            output_dir, "images", f"{pname}_{os.path.basename(image_path)}"
        )
        cv2.imwrite(crop_path, roi)

        return crop_path

    df["ImageStudyInstanceUID"] = df["image file path"].apply(StudyInstanceUID)
    df["MaskStudyInstanceUID"] = df["ROI mask file path"].apply(StudyInstanceUID)

    df["ImageSeriesInstanceUID"] = df["image file path"].apply(SeriesInstanceUID)
    df["MaskSeriesInstanceUID"] = df["ROI mask file path"].apply(SeriesInstanceUID)

    df["PatientName"] = df[["image file path", "abnormality id"]].apply(Name, axis=1)

    df["JPEGMask"] = df[
        [
            "PatientName",
            "MaskStudyInstanceUID",
            "MaskSeriesInstanceUID",
        ]
    ].apply(JPEGMask, axis=1)

    # df["JPEGCrop"] = df[
    #     [
    #         "PatientName",
    #         "MaskStudyInstanceUID",
    #         "MaskSeriesInstanceUID",
    #     ]
    # ].apply(JPEGCrop, axis=1)

    df["JPEGImage"] = df["ImageSeriesInstanceUID"].apply(JPEGImage)

    df["JPEGCrop"] = df[["JPEGImage", "JPEGMask", "PatientName"]].apply(
        JPEGCCrop, axis=1
    )

    df["mass margins"] = df["mass margins"].fillna("Unknown")
    df["mass shape"] = df["mass shape"].fillna("Unknown")

    df["assessment"] = df["assessment"].astype(str)

    df.rename(
        columns={
            "breast_density": "density",
            "mass shape": "shape",
            "mass margins": "margin",
            "assessment": "birads",
        },
        inplace=True,
    )

    if mapping and map is not None:
        df = map_columns_values(df, map)

    # Remove unnecessary columns
    df.drop(
        columns=[
            "left or right breast",
            "patient_id",
            "image view",
            "abnormality id",
            "abnormality type",
            "image file path",
            "cropped image file path",
            "ROI mask file path",
            "ImageStudyInstanceUID",
            "MaskStudyInstanceUID",
            "ImageSeriesInstanceUID",
            "MaskSeriesInstanceUID",
            "PatientName",
            "JPEGMask",
            "JPEGImage",
        ],
        inplace=True,
    )

    # Extra(s) Head
    def Malignancy(b):
        if str(b) in ["0", "4", "5"]:
            return "HIGH"
        else:
            return "LOW"

    df["malignancy"] = df["birads"].apply(Malignancy)

    # Extra(s) information
    df["DATASET"] = "cbis"

    if do_split:
        # NOTE: There is a conflict between images subtlety and what said in paper!
        # low = df[df["subtlety"] <= 2]
        # mid = df[df["subtlety"] == 3]
        # high = df[df["subtlety"] >= 4]
        low = df[df["subtlety"] >= 4]
        mid = df[df["subtlety"] == 3]
        high = df[df["subtlety"] <= 2]

        train_l, val_l, test_l = split_group(low, 0.6, 0.2, 0.2)
        train_m, val_m, test_m = split_group(mid, 0.5, 0.25, 0.25)
        train_h, val_h, test_h = split_group(high, 0.3, 0.3, 0.4)

        train = pd.concat([train_h, train_m, train_l])
        val = pd.concat([val_h, val_m, val_l])
        test = pd.concat([test_h, test_m, test_l])

        for name, split in [("train", train), ("val", val), ("test", test)]:
            split.to_csv(output_dir / f"{name}.csv")

        return {"train": train, "val": val, "test": test}
    else:
        df.to_csv(output_dir / "cbis.csv")
        return df


__all__ = ["get_cbis"]
