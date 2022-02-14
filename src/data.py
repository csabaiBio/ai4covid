from brixia.src.BSNet.model import BSNet
import numpy as np
import os
import cv2
from pathlib import Path
class ImageProcessor:
    def __init__(self, config) -> None:
        self.nets = self.__load_bsnet(config)
        self.config = config

    def __load_bsnet(self, config):
        networks = BSNet(
            seg_model_weights=os.path.join(config.weights_base_path,
                                        'segmentation-model.h5'),
            align_model_weights=os.path.join(config.weights_base_path,
                                            'alignment-model.h5'),
            bscore_model_weights=os.path.join(
                config.weights_base_path,
                'fpn_4lev_fliplr_ncl_loss03_correct_feat128-16-44.h5'),
            freeze_align_model=True,
            pretrain_aligment_net=False,
            explict_self_attention=True,
            load_bscore_model=True,
            freeze_segmentation=True,
            load_align_model=True,
            backbone_name='resnet18',
            input_shape=(config.img_size, config.img_size, 1),
            input_tensor=None,
            encoder_weights=None,
            freeze_encoder=True,
            skip_connections='default',
            decoder_block_type='transpose',
            decoder_filters=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            n_upsample_blocks=5,
            upsample_rates=(2, 2, 2, 2, 2),
            classes=4,
            activation='sigmoid',
            load_seg_model=True)
        return {
            "segmentation_model": networks[0],
            "alignment_model": networks[1],
            "score_model": networks[2]
        }

    def load_image(self, path):
        img = cv2.imread(path, 0)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (self.config.img_size, self.config.img_size),
                        interpolation=cv2.INTER_AREA)
        return img

    def process_image(self, path):
        name = Path(path).stem
        image = self.load_image(path)
        image = np.array(image) / 255.
        aligned_image = self.nets["alignment_model"].predict(
            np.expand_dims(image, [0, -1]))
        cv2.imwrite(os.path.join(self.config.output_base_path, f'{name}_aligned_img.png'), aligned_image[0] * 255.)
        segmented_image = self.nets["segmentation_model"].predict(aligned_image)
        cv2.imwrite(os.path.join(self.config.output_base_path, f'{name}_segm_img.png'), segmented_image[0] * 255.)
        return {
            "aligned_image" : aligned_image[0],
            "segmented_image": segmented_image[0]
        }