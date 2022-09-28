import os
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf

from src.data import generate_data, generate_test_data
from src.image_model import build_model

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_inference(chkpt_dir: str, save_model: bool, test: bool):
    config_path = Path(chkpt_dir) / "config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    model = build_model(config=config)

    checkpoint_path = Path(chkpt_dir) / "cp.ckpt"
    model.load_weights(checkpoint_path).expect_partial()

    prediction_model = tf.keras.models.Model(
        [
            model.get_layer(name="image").input,
            model.get_layer(name="fourier").input,
            model.get_layer(name="mask").input,
        ],
        model.get_layer(name="prognosis_out").output,
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
        try:
            config.cross_val_train = True
            datasets = generate_data(
                config, int(open(os.path.join(chkpt_dir, "fold"), "r").readline())
            )
            test_dataset, test_images = (
                datasets["validation_dataset"],
                datasets["validation_image"],
            )
        except ValueError:
            print("Full training cannot be used for CV validation.")
            exit(-1)

    test_predictions = prediction_model.predict(test_dataset)

    test_predictions = np.argmax(test_predictions, axis=-1)

    df = pd.DataFrame(columns=["file", "prognosis"])
    df["file"] = test_images
    df["prognosis"] = ["MILD" if test == 0 else "SEVERE" for test in test_predictions]

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
        os.path.join(chkpt_dir, "pred.csv" if test else "pred_valid.csv"), index=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt_dir",
        type=str,
        default=None,
    )

    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()
    run_inference(args.chkpt_dir, args.save_model, args.test)
