import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf

from src.attention_model import build_xplainable_model
from src.data import generate_test_data

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def plot_attention(image, attention_plot, n_features, IND):
    temp_image = np.array(cv2.imread(image))

    fig = plt.figure(figsize=(20, 20))

    for i in range(n_features):
        temp_att = np.resize(attention_plot[:, i], (8, 8))
        grid_size = 6
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.set_title(i)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap="gray", alpha=0.6, extent=img.get_extent())
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"attn_{IND}.png", dpi=100)


def run_inference(chkpt_dir: str, save_model: bool = False):
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

    test_dataset, test_images = generate_test_data(config)

    test_predictions, attentions = prediction_model.predict(test_dataset)

    df = pd.DataFrame(columns=["file", "prognosis"])
    df["file"] = test_images
    df["prognosis"] = ["MILD" if test < 0.5 else "SEVERE" for test in test_predictions]

    df.to_csv("pred_xplain.csv", index=False)

    for IND in range(len(test_images)):
        att = attentions[IND].reshape(64, 41)
        image = os.path.join(config.preprocessed_image_base_path, test_images[IND])
        plot_attention(image, att, 36, IND)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default="/mnt/ncshare/ai4covid_hackathon/raw_output/checkpoints/2022-02-17_21:31:41.429757",
    )

    args = parser.parse_args()
    run_inference(args.chkpt_dir)
