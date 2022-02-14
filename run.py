import wandb
import hydra
from src.data import ImageProcessor
import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from pathlib import Path


@hydra.main(config_path='src/conf', config_name='base')
def run(config):
    wandb.init(project=config.project, entity='elte-ai4covid')

    img_files = glob.glob(config.base_data_path + 'TrainSet/*')

    img_processor = ImageProcessor(config=config)

    for img in tqdm(img_files):
        _ = img_processor.process_image(img)


if __name__ == '__main__':
    run()