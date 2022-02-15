from brixia.src.BSNet.model import BSNet
import numpy as np
import os
import cv2
from pathlib import Path
import tensorflow as tf
class ImageProcessor:
    def __init__(self, config) -> None:
        self.nets = self.__load_bsnet(config)
        self.config = config

    def __load_bsnet(self, config):
        with tf.device('/cpu:0'):
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

    def is_inverted(self, image):
        height, width = image.shape
        corner_size = (int(height*self.config.corner_ratio), int(width*self.config.corner_ratio))
        tl = image[:corner_size[0], :corner_size[1]].flatten()
        bl = image[-corner_size[0]:, :corner_size[1]].flatten()
        tr = image[:corner_size[0], -corner_size[1]:].flatten()
        br = image[-corner_size[0]:, -corner_size[1]:].flatten()
        if self.config.use_bottom:
            corners = np.hstack([tl, bl, tr, br])
        else:
            corners = np.hstack([tl, tr])
                
        center = image[
            int(height*(0.5-self.config.center_ratio)):int(height*(0.5+self.config.center_ratio)),
            int(width*(0.5-self.config.center_ratio)):int(width*(0.5+self.config.center_ratio))
        ]
        return center.mean() < corners.mean()

    def load_image(self, path):
        img = cv2.imread(path, 0)
        return img

    def normlize_image(self, image):
        img = cv2.equalizeHist(image)
        img = cv2.resize(img, (self.config.img_size, self.config.img_size),
                        interpolation=cv2.INTER_AREA)
        return img

    def save_images(self, images, name, is_inverted):
        Path(os.path.join(self.config.output_base_path, 'aligned')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.config.output_base_path, 'segmented')).mkdir(parents=True, exist_ok=True)

        segmented_image = images["segmented_image"]
        cv2.imwrite(os.path.join(self.config.output_base_path, 'segmented', f'{name}_segm_img.png' if not is_inverted else f'inverted_{name}_segm_img.png'), (segmented_image * 255.).astype('uint8'))

        aligned_image = images["aligned_image"]
        cv2.imwrite(os.path.join(self.config.output_base_path, 'aligned', f'{name}_aligned_img.png' if not is_inverted else f'inverted_{name}_aligned_img.png'), (aligned_image * 255.).astype('uint8'))

    def process_image(self, path):
        name = Path(path).stem
        image = self.load_image(path)
        inverted = False
        if self.is_inverted(image):
            inverted = True
            image = np.max(image) - image
        image = self.normlize_image(image)
        image = np.array(image) / 255.
        with tf.device('/cpu:0'):
            aligned_image = self.nets["alignment_model"].predict(
                np.expand_dims(image, [0, -1]))
        with tf.device('/cpu:0'):
            segmented_image = self.nets["segmentation_model"].predict(aligned_image)
        
        output = {
            "aligned_image" : aligned_image[0],
            "segmented_image": segmented_image[0]
        }
        
        if self.config.save_images:
            self.save_images(output, name, inverted)

        return output