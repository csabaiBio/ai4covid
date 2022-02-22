import os
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf

from src.base_model import build_model
from src.data import generate_test_data

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_inference(chkpt_dir: str, save_model: bool = False):
    config_path = Path(chkpt_dir) / "config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    model = build_model(config=config)

    checkpoint_path = Path(chkpt_dir) / "cp.ckpt"
    model.load_weights(checkpoint_path).expect_partial()

    prediction_model = tf.keras.models.Model(
        [
            model.get_layer(name="image").input,
            model.get_layer(name="mask").input,
            model.get_layer(name="meta").input,
        ],
        model.get_layer(name="prognosis_out").output,
    )

    prediction_model.summary()

    if save_model:
        tf.saved_model.save(
            prediction_model,
            os.path.join(chkpt_dir, "saved_model"),
        )

    test_dataset, test_images = generate_test_data(config)

    test_predictions = prediction_model.predict(test_dataset)

    test_predictions = np.argmax(test_predictions, axis=-1)

    df = pd.DataFrame(columns=["file", "prognosis"])
    df["file"] = test_images
    df["prognosis"] = ["MILD" if test == 0 else "SEVERE" for test in test_predictions]

    df.to_csv("pred.csv", index=False)


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
