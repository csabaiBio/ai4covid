from brixia.src.BSNet.model import BSNet
from PIL import Image, ImageOps, ImageStat
import numpy as np
import os
import cv2


def load_bsnet(config):
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


def load_image(path, config):
    img = cv2.imread('/home/qbeer/GitHub/ai4covid/data/TrainSet/P_1_10.png', 0)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (config.img_size, config.img_size),
                     interpolation=cv2.INTER_AREA)
    return img
