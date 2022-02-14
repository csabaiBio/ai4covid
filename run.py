import wandb
import hydra
from src.data import load_bsnet, load_image
import glob
from tqdm import tqdm
import os
import numpy as np
import cv2


@hydra.main(config_path='src/conf', config_name='base')
def run(config):
    wandb.init(project=config.project, entity='elte-ai4covid')

    img_files = glob.glob(config.base_data_path + 'TrainSet/*')

    for img in tqdm(img_files):
        print(img)
        image = load_image(img, config)
        cv2.imwrite(os.path.join(config.output_base_path, 'img.png'), image)
        image = np.array(image) / 255.
        nets = load_bsnet(config)
        predictions = nets["alignment_model"].predict(
            np.expand_dims(image, [0, -1]))
        cv2.imwrite(os.path.join(config.output_base_path, 'aligned_img.png'), predictions[0] * 255.)
        predictions = nets["segmentation_model"].predict(predictions)
        cv2.imwrite(os.path.join(config.output_base_path, 'segm_img.png'), predictions[0] * 255.)
        break


if __name__ == '__main__':
    run()