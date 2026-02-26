import json
from typing import List, Dict, Union, Literal, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def create_onehot(index: int, num_classes: int) -> List[int]:
    one_hot = [0] * num_classes
    one_hot[index] = 1
    return one_hot


def map_columns_values(
    df: pd.DataFrame,
    map: Dict[str, Union[str, Dict[Literal["action"], str]]],
) -> pd.DataFrame:
    map = map if not isinstance(map, str) else json.load(open(map, "r"))
    for column in map:
        target_map = map[column]

        assert isinstance(
            target_map, dict
        ), f"mapping key must be a dict not {type(target_map)}"

        replacments = {
            key: value for key, value in target_map.items() if isinstance(value, str)
        }
        actions = {
            key: value for key, value in target_map.items() if isinstance(value, dict)
        }

        df[column] = df[column].replace(replacments)

        for label, action in actions.items():
            for action_key, action_value in action.items():
                if action_key == "action":
                    if action_value == "drop":
                        df = df.drop(df[df[column] == label].index)
    return df


def split_group(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state=42,
    stratify: str = "pathology",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert train_frac + val_frac + test_frac == 1.0

    train, temp = train_test_split(
        df,
        test_size=(1 - train_frac),
        stratify=df[stratify],
        random_state=random_state,
    )

    val, test = train_test_split(
        temp,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp[stratify],
        random_state=random_state,
    )

    return train, val, test


__all__ = ["create_onehot", "map_columns_values", "split_group"]
