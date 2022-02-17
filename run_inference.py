import os
import random
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.base_model import build_model
from src.image_proc import ImageProcessor
from src.table_proc import TableProcessor

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@hydra.main(config_path="src/conf", config_name="test")
def run_inference(chkpt_dir: str, save_model: bool):
    config_path = Path(chkpt_dir) / "config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    model = build_model(config=config)

    checkpoint_path = Path(chkpt_dir) / "cp.ckpt"
    model.load_weights(checkpoint_path).expect_partial()

    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="model_output").output
    )

    prediction_model.summary()

    if save_model:
        tf.saved_model.save(
            prediction_model,
            os.path.join(chkpt_dir, "saved_model"),
        )

    img_processor = ImageProcessor(config=config)
    table_prcessor = TableProcessor(config=config)

    df = pd.read_excel(config.test_table_path)
    df.drop(["Death", "Prognosis"], axis=1, inplace=True)

    for idx in tqdm(range(len(df))):
        row = df.iloc[[idx]]
        image_file, _ = table_prcessor.line_impute_population_average(row)
        img_path = os.path.join(config.image_base_path, image_file)
        _ = img_processor.process_image(img_path)


if __name__ == "__main__":
    run_inference()
