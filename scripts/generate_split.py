import argparse
import os
from ast import arg
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(args):
    df = pd.read_csv(args.base_csv)
    csv_name = Path(args.base_csv).stem
    base_csv_path = Path(args.base_csv).parent
    for i in range(args.n_splits):
        train, test = train_test_split(df, test_size=args.test_size, random_state=13742)
        train.to_csv(
            os.path.join(base_csv_path, csv_name + f"_cv{i + 1}.csv"), index=False
        )
        test.to_csv(
            os.path.join(
                base_csv_path, csv_name.replace("train", "valid") + f"_cv{i + 1}.csv"
            ),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_csv",
        type=str,
        default="./data/preprocessed_data/tables/pop_avg/trainClinDataImputedPopAvg.csv",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()

    split(args)
