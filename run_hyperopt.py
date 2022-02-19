import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import tensorflow as tf
from omegaconf import DictConfig
from tqdm import tqdm
from wandb.keras import WandbCallback

import wandb
from src.base_model import build_model
from src.data import generate_data

np.random.seed(42)
tf.random.set_seed(137)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(13742)

tf.keras.backend.clear_session()
tf.autograph.set_verbosity(level=0, alsologtostdout=False)
tf.get_logger().setLevel(3)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


@hydra.main(config_path="src/conf", config_name="train_pop_sampled")
def run_experiments(config: DictConfig):
    wandb.init(project=config.project, entity="elte-ai4covid")
    datasets = generate_data(config, fold=np.random.randint(0, 5))

    model = build_model(config)

    chkpt_dir = (
        Path(config.raw_output_base)
        / "checkpoints"
        / datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    )

    Path(chkpt_dir).mkdir(exist_ok=True, parents=True)
    chkpt_path = Path(chkpt_dir) / "cp.ckpt"

    omegaconf.OmegaConf.save(config=config, f=Path(chkpt_dir) / "config.yaml")

    model.summary()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpt_path,
        save_weights_only=True,
        verbose=0,
        monitor="val_loss",
        save_freq="epoch",
        mode="min",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=3,
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )

    _ = model.fit(
        datasets["train_dataset"],
        validation_data=datasets["validation_dataset"],
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=[WandbCallback(), cp_callback, early_stopping],
        verbose=0,
    )


if __name__ == "__main__":
    run_experiments()
