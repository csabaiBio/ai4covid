import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.attention_model import build_xplainable_model
from src.data import generate_data, generate_test_data


def plot_attention(image, attention_plot, n_features, IND, config):
    if type(image) == np.ndarray:
        temp_image = image
    else:
        temp_image = np.array(cv2.imread(image))

    fig = plt.figure(figsize=(20, 20))

    for i in range(n_features):
        temp_att = np.resize(attention_plot[:, i], (16, 16))
        ax = fig.add_subplot(4, 5, i + 1)
        ax.set_title(config.datasets[config.dataset_identifier].feature_cols[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap="gray", alpha=0.6, extent=img.get_extent())
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_path = os.path.join(config.raw_output_base, "attentions", f"attn_{IND}.png")
    Path(output_path).parents[0].mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


def run_inference(chkpt_dir: str, save_model: bool, test: bool, plot_all: bool):
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

    if save_model:
        tf.saved_model.save(
            prediction_model,
            os.path.join(chkpt_dir, "saved_model"),
        )

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

    mean_attention = np.mean(attentions, axis=0)
    std_attention = np.std(attentions, axis=0)

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size)),
        mean_attention,
        20,
        "mean" if test else "valid_mean",
        config,
    )

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size)),
        std_attention,
        20,
        "std" if test else "valid_std",
        config,
    )

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size)),
        np.log(mean_attention),
        20,
        "log_mean" if test else "valid_log_mean",
        config,
    )

    plot_attention(
        np.zeros(shape=(config.img_size, config.img_size)),
        np.log(std_attention),
        20,
        "log_std" if test else "valid_log_std",
        config,
    )

    if args.plot_all:
        for IND in tqdm(range(len(test_images))):
            att = attentions[IND].reshape(256, 20)
            image = os.path.join(
                config.preprocessed_image_base_path.replace(
                    "train", "test" if test else "train"
                ),
                test_images[IND],
            )
            plot_attention(image, att, 20, IND, config)

    df = pd.DataFrame(columns=["file", "prognosis"])
    df["file"] = test_images
    df["prognosis"] = [
        "MILD" if test < 0.5 else "SEVERE" for test in test_predictions.flatten()
    ]

    df.to_csv("pred_xplain.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default="/mnt/ncshare/ai4covid_hackathon/raw_output/checkpoints/2022-02-17_21:31:41.429757",
    )

    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--plot_all", action="store_true", default=False)

    args = parser.parse_args()
    run_inference(args.chkpt_dir, args.save_model, args.test, args.plot_all)
