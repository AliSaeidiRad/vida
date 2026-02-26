import math
from pathlib import Path
from typing import List, Dict, Union, Literal, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from masscls.utils.dataset import get_cbis, get_vida, get_csaw, get_vindr, get_cesm


def get_dataset(
    name: str,
    extra: Dict[str, Any],
    output: Union[Path, str],
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    output = output if isinstance(output, Path) else Path(output)
    if not output.exists():
        output.mkdir()

    if name == "cbis":
        return get_cbis(
            csv=extra["csv"],
            jpeg=extra["jpeg"],
            map=extra["map"],
            output_dir=output,
        )  # type: ignore
    elif name == "vida":
        return get_vida(
            images=extra["images"],
            annotations=extra["annotations"],
            map=extra["map"],
            output_dir=output,
        )
    elif name == "csaw":
        return get_csaw(
            images=extra["images"],
            masks=extra["masks"],
            screening_data=extra["screening_data"],
            output_dir=output,
        )
    elif name == "vindr":
        return get_vindr(
            images=extra["images"],
            finding=extra["finding"],
            output_dir=output,
        )
    elif name == "cdd-cesm":
        return get_cesm(
            annotations=extra["annotations"],
            segmentations=extra["segmentations"],
            images=extra["images"],
            map=extra["map"],
            output_dir=output,
        )
    else:
        raise ValueError(f"dataset `{name}` is not supported!")


def prepare_dataset(
    datasets: Dict[str, Dict[str, Any]], output: Union[str, Path]
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    output = output if isinstance(output, Path) else Path(output)

    if not output.exists():
        output.mkdir()

    train = []
    val = []
    test = []

    for dataset in datasets:
        output_dataset = output / dataset
        if (
            not output_dataset.exists()
            or not output_dataset.joinpath("train.csv").exists()
        ):
            output_dataset.mkdir(exist_ok=True)
            dfs = get_dataset(dataset, datasets[dataset], output_dataset)
            train.append(dfs["train"])
            val.append(dfs["val"])
            test.append(dfs["test"])
        else:
            train.append(pd.read_csv(output_dataset / "train.csv"))
            val.append(pd.read_csv(output_dataset / "val.csv"))
            test.append(pd.read_csv(output_dataset / "test.csv"))

    return {"train": pd.concat(train), "val": pd.concat(val), "test": pd.concat(test)}


def plot_dataset_distribution(
    dfs: Dict[Literal["train", "test", "val"], pd.DataFrame],
    categorical_cols: List[str],
):
    train_df = dfs["train"]
    val_df = dfs["val"]
    test_df = dfs["test"]

    total_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    n = len(categorical_cols)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(6 * ncols, 4 * nrows))

    for idx, col in enumerate(categorical_cols):
        df_counts = pd.DataFrame({"Class": total_df[col].unique()})
        df_counts["Total"] = df_counts["Class"].apply(
            lambda x: (total_df[col] == x).sum()
        )
        df_counts["Train"] = df_counts["Class"].apply(
            lambda x: (train_df[col] == x).sum()
        )
        df_counts["Val"] = df_counts["Class"].apply(lambda x: (val_df[col] == x).sum())
        df_counts["Test"] = df_counts["Class"].apply(
            lambda x: (test_df[col] == x).sum()
        )
        df_long = df_counts.melt(
            id_vars="Class",
            value_vars=["Total", "Train", "Val", "Test"],
            var_name="Split",
            value_name="Count",
        )

        ax = fig.add_subplot(nrows, ncols, idx + 1)
        sns.barplot(data=df_long, x="Class", y="Count", hue="Split", ax=ax)
        for container in ax.containers:
            ax.bar_label(
                container,  # type: ignore
                fmt="%d",
                padding=1,
                fontsize=8,
            )

        ax.set_title(f"Distribution of {col} by Dataset Split")
        ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    return fig


__all__ = [
    "get_dataset",
    "prepare_dataset",
    "plot_dataset_distribution",
]
