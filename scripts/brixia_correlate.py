from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
from skimage.transform import resize

cols = [
    "Age",
    "Sex",
    "PositivityAtAdmission",
    "Temp_C",
    "DaysFever",
    "Cough",
    "DifficultyInBreathing",
    "WBC",
    "RBC",
    "CRP",
    "Fibrinogen",
    "Glucose",
    "PCT",
    "LDH",
    "INR",
    "D_dimer",
    "Ox_percentage",
    "PaO2",
    "SaO2",
    "PaCO2",
    "pH",
    "CardiovascularDisease",
    "IschemicHeartDisease",
    "AtrialFibrillation",
    "HeartFailure",
    "Ictus",
    "HighBloodPressure",
    "Diabetes",
    "Dementia",
    "BPCO",
    "Cancer",
    "ChronicKidneyDisease",
    "RespiratoryFailure",
    "Obesity",
    "Position",
]


def mean_interpolate(arr):
    rescaled = []
    for _arr in arr:
        out = resize(_arr, (16, 16), preserve_range=True)
        rescaled.append(out)
    br_map = np.mean(np.array(rescaled), axis=0)
    br_map = (br_map - np.mean(br_map)) / np.std(br_map)
    return br_map


def reshape(inp):
    return np.moveaxis(inp.reshape(16, 16, 35), 2, 0)


def run(chkpt_dir: str):
    test_mean_mild_map = np.load(f"{chkpt_dir}/attentions/mean_mild_attentions.npy")
    test_mean_severe_map = np.load(f"{chkpt_dir}/attentions/mean_severe_attentions.npy")

    mean_mild_map = np.load(f"{chkpt_dir}/attentions/valid_mean_mild_attentions.npy")
    mean_severe_map = np.load(
        f"{chkpt_dir}/attentions/valid_mean_severe_attentions.npy"
    )

    mean_mild_map = reshape(mean_mild_map)
    mean_severe_map = reshape(mean_severe_map)
    test_mean_mild_map = reshape(test_mean_mild_map)
    test_mean_severe_map = reshape(test_mean_severe_map)

    brixia_scores = np.array(
        [
            np.loadtxt(fname).reshape(3, 2).astype(int)
            for fname in Path("data/preprocessed_data/train/score_original/").glob(
                "*.txt"
            )
        ]
    )

    brixia_paths = np.array(
        [
            fname.name
            for fname in Path("data/preprocessed_data/train/score_original/").glob(
                "*.txt"
            )
        ]
    ).tolist()

    test_brixia_scores = np.array(
        [
            np.loadtxt(fname).reshape(3, 2).astype(int)
            for fname in Path("data/preprocessed_data/test/score_original/").glob(
                "*.txt"
            )
        ]
    )

    test_brixia_paths = np.array(
        [
            fname.name
            for fname in Path("data/preprocessed_data/test/score_original/").glob(
                "*.txt"
            )
        ]
    ).tolist()

    prognosis = (
        pd.read_csv("data/preprocessed_data/tables/trainClinData.csv")["Prognosis"]
        .map(lambda x: int(x == "SEVERE"))
        .values
    )
    test_prognosis = (
        pd.read_excel("data/raw_data/completeTestClinData.xls")["Prognosis"]
        .map(lambda x: int(x == "SEVERE"))
        .values
    )

    names = (
        pd.read_csv("data/preprocessed_data/tables/trainClinData.csv")["ImageFile"]
        .map(lambda x: x.replace(".png", ".txt"))
        .values.tolist()
    )
    test_names = (
        pd.read_csv("data/preprocessed_data/tables/testClinData.csv")["ImageFile"]
        .map(lambda x: x.replace(".png", ".txt"))
        .values.tolist()
    )

    indices = np.where(np.in1d(names, brixia_paths, assume_unique=True))[0]
    test_indices = np.where(np.in1d(test_names, test_brixia_paths, assume_unique=True))[
        0
    ]

    mean_severe_brixia_scores = brixia_scores[indices][np.nonzero(prognosis)]
    mean_mild_brixia_scores = brixia_scores[indices][np.nonzero(1.0 - prognosis)]

    test_mean_severe_brixia_scores = test_brixia_scores[test_indices][
        np.nonzero(test_prognosis)
    ]
    test_mean_mild_brixia_scores = test_brixia_scores[test_indices][
        np.nonzero(1.0 - test_prognosis)
    ]

    test_mean_severe_brixia_scores = mean_interpolate(test_mean_severe_brixia_scores)
    test_mean_mild_brixia_scores = mean_interpolate(test_mean_mild_brixia_scores)

    mean_severe_brixia_scores = mean_interpolate(mean_severe_brixia_scores)
    mean_mild_brixia_scores = mean_interpolate(mean_mild_brixia_scores)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes = axes.flatten()

    cdata = axes[0].imshow(
        test_mean_severe_brixia_scores,
        vmin=-1.5,
        vmax=1.5,
        interpolation="none",
        cmap="coolwarm",
    )
    axes[0].set_title("TEST, SEVERE")
    cbar = plt.colorbar(cdata, ax=axes[0])
    cbar.minorticks_on()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    cdata = axes[1].imshow(
        test_mean_mild_brixia_scores,
        vmin=-1.5,
        vmax=1.5,
        interpolation="none",
        cmap="coolwarm",
    )
    axes[1].set_title("TEST, MILD")
    cbar = plt.colorbar(cdata, ax=axes[1])
    cbar.minorticks_on()
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    cdata = axes[2].imshow(
        test_mean_severe_brixia_scores - test_mean_mild_brixia_scores,
        vmin=-0.1,
        vmax=0.1,
        interpolation="none",
        cmap="coolwarm",
    )
    axes[2].set_title("TEST, SEVERE - MILD")
    cbar = plt.colorbar(cdata, ax=axes[2])
    cbar.minorticks_on()
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    cdata = axes[3].imshow(
        mean_severe_brixia_scores,
        vmin=-1.5,
        vmax=1.5,
        interpolation="none",
        cmap="coolwarm",
    )
    axes[3].set_title("VALID, SEVERE")
    cbar = plt.colorbar(cdata, ax=axes[3])
    cbar.minorticks_on()
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    cdata = axes[4].imshow(
        mean_mild_brixia_scores,
        vmin=-1.5,
        vmax=1.5,
        interpolation="none",
        cmap="coolwarm",
    )
    axes[4].set_title("VALID, MILD")
    cbar = plt.colorbar(cdata, ax=axes[4])
    cbar.minorticks_on()
    axes[4].set_xticks([])
    axes[4].set_yticks([])

    cdata = axes[5].imshow(
        mean_severe_brixia_scores - mean_mild_brixia_scores,
        vmin=-0.1,
        vmax=0.1,
        interpolation="none",
        cmap="coolwarm",
    )
    axes[5].set_title("VALID, SEVERE - MILD")
    cbar = plt.colorbar(cdata, ax=axes[5])
    cbar.minorticks_on()
    axes[5].set_xticks([])
    axes[5].set_yticks([])

    fig.tight_layout()
    plt.savefig(Path(chkpt_dir) / "attentions/brixia-comparison.png", dpi=100)

    def correlation_study(arr, brixia, cols, threshold=0.0):
        for _arr, _col in zip(arr, cols):
            corr = np.correlate(_arr.flatten(), brixia.flatten())[0]
            if np.abs(corr) > threshold:
                print(f"{_col:25s} : {corr:10.4f}")

    print("\nTEST")
    print("TEST - MILD")
    correlation_study(test_mean_mild_map, test_mean_mild_brixia_scores, cols, 0.1)
    print("\nTEST - SEVERE")
    correlation_study(test_mean_severe_map, test_mean_severe_brixia_scores, cols, 0.1)
    print("\nTEST, SEVERE - MILD")
    correlation_study(
        test_mean_severe_map - test_mean_mild_map,
        test_mean_severe_brixia_scores - test_mean_mild_brixia_scores,
        cols,
        0.0025,
    )

    print("\n\nVALID")
    print("VALID - MILD")
    correlation_study(mean_mild_map, mean_mild_brixia_scores, cols, 0.1)
    print("\nVALID - SEVERE")
    correlation_study(mean_severe_map, mean_severe_brixia_scores, cols, 0.1)
    print("\nVALID, SEVERE - MILD")
    correlation_study(
        mean_severe_map - mean_mild_map,
        mean_severe_brixia_scores - mean_mild_brixia_scores,
        cols,
        0.0025,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default="/mnt/ncshare/ai4covid_hackathon/raw_output/checkpoints/2022-02-17_21:31:41.429757",
    )

    args = parser.parse_args()
    run(
        args.chkpt_dir,
    )
