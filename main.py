import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        default="train",
        help="",
    )
    parser.add_argument("--datasets")
