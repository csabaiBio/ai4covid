import os
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from src.attention_model import build_xplainable_model
from src.data import generate_data, generate_test_data

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def plot_attention(
    image,
    attention_plot,
    IND,
    config,
    chkpt_dir,
    threshold=False,
    only_correlating_cols=False,
    global_threshold=False,
):
    if type(image) == np.ndarray:
        temp_image = image
    else:
        temp_image = np.array(cv2.imread(image))

    n_features = len(config.datasets[config.dataset_identifier].feature_cols)

    fig = plt.figure(figsize=(48, 12))

    if only_correlating_cols:
        col_indicies = [5, 8, 10, 11, 14, 16, 18, 19, 20, 21, 22, 25, 26, 27, 29, 31]
        features = list(
            enumerate(config.datasets[config.dataset_identifier].feature_cols)
        )
        features_to_keep = list(
            filter(
                lambda x: x[1]
                not in ["ImageFile", "Prognosis", "Death", "Age", "Sex", "Position"],
                features,
            )
        )
        features_to_keep = list(
            filter(
                lambda x: x[1]
                not in [
                    f"Hospital_{letter}" for letter in ["A", "B", "C", "D", "E", "F"]
                ],
                features_to_keep,
            )
        )
        n_features = len(features_to_keep)
        j = 0

    global_minimum = np.min(attention_plot)
    global_maximum = np.max(attention_plot)

    if only_correlating_cols:
        attentions = []
        for feat in features_to_keep:
            if feat[0] not in col_indicies:
                attentions.append(attention_plot[:, feat[0]])
        attentions = np.array(attentions)
        global_minimum = np.min(attentions)
        global_maximum = np.max(attentions)

    print(f"Global minimum : {global_minimum}, global maximum : {global_maximum}")

    for i in range(n_features):
        if only_correlating_cols:
            if not (features_to_keep[i][0] in col_indicies):
                temp_att = np.resize(
                    attention_plot[:, features_to_keep[i][0]], (16, 16)
                )
                minimum = np.min(temp_att)
                maximum = np.max(temp_att)

                if threshold and not global_threshold:
                    if minimum < 0:
                        temp_att /= np.abs(minimum)
                    else:
                        temp_att -= minimum
                        temp_att /= maximum - minimum
                    minimum = np.min(temp_att)
                    maximum = np.max(temp_att)

                if global_threshold:
                    if global_minimum < 0:
                        temp_att /= np.abs(global_minimum)
                    else:
                        temp_att -= global_minimum
                        temp_att /= global_maximum - global_minimum
                    minimum = -1 if global_minimum < 0 else 0
                    maximum = 1

                ax = fig.add_subplot(2, 8, j + 1)
                title = "\n".join(textwrap.wrap(features_to_keep[i][1], 7))
                ax.set_title(title, fontsize=55)
                img = ax.imshow(temp_image, cmap="gray", vmin=0, vmax=1)
                cbar_data = ax.imshow(
                    temp_att,
                    cmap="coolwarm" if minimum < 0 else "gist_heat",
                    alpha=0.6,
                    vmin=minimum,
                    vmax=maximum,
                    extent=img.get_extent(),
                )
                cbar = fig.colorbar(cbar_data, ax=ax, extend="both")
                cbar.set_ticks([])
                ax.set_xticks([])
                ax.set_yticks([])
                j += 1
        else:
            temp_att = np.resize(attention_plot[:, i], (16, 16))
            minimum = np.min(temp_att)
            maximum = np.max(temp_att)

            if threshold and not global_threshold:
                if minimum < 0:
                    temp_att /= np.abs(minimum)
                else:
                    temp_att -= minimum
                    temp_att /= maximum - minimum
                minimum = np.min(temp_att)
                maximum = np.max(temp_att)

            if global_threshold:
                if global_minimum < 0:
                    temp_att /= np.abs(global_minimum)
                else:
                    temp_att -= global_minimum
                    temp_att /= global_maximum - global_minimum
                minimum = -1 if global_minimum < 0 else 0
                maximum = 1

            ax = fig.add_subplot(3, 12, i + 1)
            title = config.datasets[config.dataset_identifier].feature_cols[i]
            if len(title) > 8:
                title = "".join([c if c.isupper() else "" for c in title])
            if len(title) == 1:
                title = config.datasets[config.dataset_identifier].feature_cols[i][:8]
            ax.set_title(title, fontsize=50)
            img = ax.imshow(temp_image, cmap="gray", vmin=0, vmax=1)
            cbar_data = ax.imshow(
                temp_att,
                cmap="coolwarm" if minimum < 0 else "gist_heat",
                alpha=0.6,
                vmin=minimum,
                vmax=-minimum if minimum < 0 else maximum,
                extent=img.get_extent(),
            )
            cbar = fig.colorbar(
                cbar_data,
                ax=ax,
                extend="both",
                ticks=np.linspace(
                    minimum, -minimum if minimum < 0 else maximum, 9, endpoint=True
                ),
            )
            cbar.set_ticks([])
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_path = os.path.join(
        chkpt_dir,
        "attentions",
        "corr" if only_correlating_cols else "all",
        "thresholded"
        if threshold
        else "non_thresholded"
        if not global_threshold
        else "global_threshold",
        f"attn_{IND}.png",
    )
    Path(output_path).parents[0].mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


def run_inference(
    chkpt_dir: str,
    test: bool,
    plot_all: bool,
    threshold: bool,
    only_correlating_cols: bool,
    global_threshold: bool,
):
    config_path = Path(chkpt_dir) / "config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    model = build_xplainable_model(config=config)

    checkpoint_path = Path(chkpt_dir) / "cp.ckpt"
    model.load_weights(checkpoint_path).expect_partial()

    prediction_model = tf.keras.models.Model(
        [
            model.get_layer(name="image").input,
            model.get_layer(name="mask").input,
            model.get_layer(name="meta").input,
        ],
        [
            model.get_layer(name="prognosis_out").output,
            model.get_layer(name="attention_weights").output,
        ],
    )

    prediction_model.summary()

    if test:
        test_dataset, test_images = generate_test_data(config)
    else:
        config.cross_val_train = True
        datasets = generate_data(
            config, int(open(os.path.join(chkpt_dir, "fold"), "r").readline())
        )
        test_dataset, test_images = (
            datasets["validation_dataset"],
            datasets["validation_image"],
        )

    test_predictions, attentions = prediction_model.predict(test_dataset)

    D = np.load("diff.npy")
    r = np.load("ratio.npy")

    b = D / (r + 1)
    a = r * b

    min_a, max_a = np.min(a), np.max(a)
    min_b, max_b = np.min(b), np.max(b)
    # min-max scale
    a -= min_a
    a /= max_a - min_a

    b -= min_b
    b /= max_b - min_b

    print(f"min_a : {np.min(a)} | max_a : {np.max(a)}")
    print(f"min_b : {np.min(b)} | max_b : {np.max(b)}")

    discretized_test = test_predictions
    discretized_test[discretized_test >= 0.5] = 1
    discretized_test[discretized_test < 0.5] = 0

    severe_ind = np.nonzero(discretized_test)[0]
    mild_ind = np.nonzero(discretized_test - 1)[0]

    mean_mild = np.mean(attentions[mild_ind], axis=0)
    mean_severe = np.mean(attentions[severe_ind], axis=0)

    output_path = os.path.join(
        chkpt_dir,
        "attentions",
        "mean_severe_attentions.npy" if test else "valid_mean_severe_attentions.npy",
    )
    Path(output_path).parents[0].mkdir(exist_ok=True, parents=True)
    np.save(output_path, mean_severe)
    np.save(
        output_path.replace("mean_severe_attentions", "mean_mild_attentions"), mean_mild
    )

    all_output_path = os.path.join(
        chkpt_dir,
        "attentions",
        "attentions.npy" if test else "valid_attentions.npy",
    )
    np.save(all_output_path, attentions)

    pred_output_path = os.path.join(
        chkpt_dir,
        "attentions",
        "predictions.npy" if test else "valid_predictions.npy",
    )
    np.save(pred_output_path, test_predictions)

    plot_attention(
        a,
        mean_mild,
        "a_mean_mild" if test else "valid_a_mean_mild",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        b,
        mean_mild,
        "b_mean_mild" if test else "valid_b_mean_mild",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        a,
        mean_severe,
        "a_mean_severe" if test else "valid_a_mean_severe",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        b,
        mean_severe,
        "b_mean_severe" if test else "valid_b_mean_severe",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size, 1)),
        mean_severe,
        "mean_severe" if test else "valid_mean_severe",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size, 1)),
        mean_mild,
        "mean_mild" if test else "valid_mean_mild",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size, 1)),
        mean_severe - mean_mild,
        "mean_severe_minus_mild" if test else "valid_mean_severe_minus_mild",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        a,
        mean_severe - mean_mild,
        "a_mean_severe_minus_mild" if test else "valid_a_mean_severe_minus_mild",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    plot_attention(
        b,
        mean_severe - mean_mild,
        "b_mean_severe_minus_mild" if test else "valid_b_mean_severe_minus_mild",
        config,
        chkpt_dir,
        threshold,
        only_correlating_cols,
        global_threshold,
    )

    if args.plot_all:
        for IND in tqdm(range(len(test_images))):
            att = attentions[IND].reshape(256, -1)
            image = os.path.join(
                config.preprocessed_image_base_path.replace(
                    "train", "test" if test else "train"
                ),
                test_images[IND],
            )
            plot_attention(
                image,
                att,
                IND,
                config,
                chkpt_dir,
                threshold,
                only_correlating_cols,
                global_threshold,
            )

    df = pd.DataFrame(columns=["file", "prognosis"])
    df["file"] = test_images
    df["prognosis"] = [
        "MILD" if test < 0.5 else "SEVERE" for test in test_predictions.flatten()
    ]

    if test:
        whole_test_df = pd.read_excel(
            "/mnt/ncshare/ai4covid_hackathon/raw_data/completeTestClinData.xls"
        )
        test_raw_df = whole_test_df[["ImageFile", "Prognosis"]]
        df["prognosis_real"] = test_raw_df["Prognosis"].values
    else:
        dataset = config.datasets[config.dataset_identifier]
        fold = int(open(os.path.join(chkpt_dir, "fold"), "r").readline()) + 1
        actual = pd.read_csv(dataset.cv_valid_table + f"cv{fold}.csv")
        df["prognosis_real"] = [
            "MILD" if val in (0, "MILD") else "SEVERE" for val in actual["Prognosis"]
        ]

    df.to_csv(
        os.path.join(chkpt_dir, "pred_xplain.csv" if test else "pred_xplain_valid.csv"),
        index=False,
    )

    print(
        f'Balanced accuracy : {balanced_accuracy_score(df["prognosis_real"].values, df["prognosis"].values)}'
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default="/mnt/ncshare/ai4covid_hackathon/raw_output/checkpoints/2022-02-17_21:31:41.429757",
    )

    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--plot_all", action="store_true", default=False)
    parser.add_argument("--threshold", action="store_true", default=False)
    parser.add_argument("--only_correlating_cols", action="store_true", default=False)
    parser.add_argument("--global_threshold", action="store_true", default=False)

    args = parser.parse_args()
    run_inference(
        args.chkpt_dir,
        args.test,
        args.plot_all,
        args.threshold,
        args.only_correlating_cols,
        args.global_threshold,
    )
