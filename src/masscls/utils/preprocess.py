import os
from warnings import warn
from typing import Union, Literal, List, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm


class ISS:
    def __init__(
        self,
        percentiles: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> None:
        self.percentiles = (
            np.asarray(
                [
                    1,
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    99,
                ],
                dtype=np.float32,
            )
            if percentiles is None
            else percentiles
        )
        self.landmarks = (
            np.asarray(
                [
                    12.0,
                    24.0,
                    31.0,
                    38.0,
                    47.0,
                    58.0,
                    70.0,
                    85.0,
                    105.0,
                    135.0,
                    203.0,
                ],
                dtype=np.float32,
            )
            if landmarks is None
            else landmarks
        )

    def train(
        self,
        inputs: Dict[Literal["image", "mask"], List[Union[str, np.ndarray]]],
        percentiles: List[int],
    ):
        self.percentiles = percentiles
        all_landmarks: List[np.ndarray] = []

        for image_path, mask_path in tqdm(
            zip(inputs["image"], inputs["mask"]),
            desc="Processing",
            total=len(inputs["image"]),
            leave=False,
        ):
            if isinstance(image_path, str) and isinstance(mask_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            elif isinstance(image_path, np.ndarray) and isinstance(
                mask_path, np.ndarray
            ):
                image = image_path
                mask = mask_path
            else:
                raise ValueError("inputs has wrong data types")

            if image is None or mask is None:
                warn(f"`{image}` or `{mask}` is corrupted`")
                continue

            img = image.astype(np.float32)
            mask = mask.astype(np.float32)

            if mask.max() > 1:
                mask /= 255.0

            roi = img[mask > 0]  # breast region only
            if roi.size == 0:
                continue

            landmarks = np.percentile(roi, percentiles).astype(np.float32)
            all_landmarks.append(landmarks)

        self.landmarks = np.median(np.stack(all_landmarks, axis=0), axis=0)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[Union[np.ndarray, bool]] = None,
    ):
        assert self.landmarks is not None
        assert self.percentiles is not None

        image.astype(np.float32)

        if isinstance(mask, bool):
            mask = self.remove_background(image)

        if mask is not None:
            mask = mask.astype(np.float32)
            if mask.max() > 1:
                mask /= 255.0
            roi = image[mask > 0]
        else:
            roi = image

        img_perc = np.percentile(roi, self.percentiles)
        mapped = np.interp(image, img_perc, self.landmarks)
        mapped = np.clip(mapped, 0, 255).astype(np.uint8)
        if mask is not None:
            mapped = mapped * np.expand_dims(mask.astype(np.uint8), -1)

        return mapped

    @staticmethod
    def remove_background(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.uint8)
        # Binarize (assumes background is darker or equal thresholding works for your case)
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

        kernel = np.ones((250, 250), np.uint8)

        # Erode to remove small connections/noise and simplify regions
        eroded = cv2.erode(threshold, kernel, iterations=1)

        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply mask to keep only selected regions
        kept = cv2.bitwise_and(threshold, threshold, mask=mask)

        # Dilate to restore the eroded regions' size
        cleaned = cv2.dilate(kept, kernel, iterations=1)

        return cleaned
