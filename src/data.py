import pandas as pd
import tensorflow as tf
from albumentations import Compose, GaussianBlur, RandomCrop, Resize, Rotate
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

def aug_fn(image, augment, width, height):
    data = {"image": image}
    if augment:
        transforms = Compose(
            [
                RandomCrop(
                    int(width - 0.05 * width), int(height - 0.05 * height), p=0.5
                ),
                Rotate(limit=10),
                GaussianBlur(sigma_limit=1.2),
                Resize(
                    width,
                    height,
                    always_apply=True,
                ),
            ]
        )
    else:
        transforms = Compose(
            [
                Resize(
                    width,
                    height,
                    always_apply=True,
                )
            ]
        )

    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    return aug_img

def process_data(sample, augment, img_width, img_height):
    image = sample["image"]
    aug_img = tf.numpy_function(
        func=aug_fn,
        inp=[image, augment, img_width, img_height],
        Tout=tf.float32,
    )
    sample["image"] = aug_img
    return sample


def set_shapes(sample, config):
    img = sample["image"]
    mask = sample["mask"]
    meta = sample["meta"]
    prognosis = sample["prognosis"]
    death = sample["death"]

    img.set_shape((config.img_width, config.img_height, 1))
    mask.set_shape((config.img_width, config.img_height, 1))
    meta.set_shape((config.n_feature_cols))
    prognosis.set_shape(())
    death.set_shape(())

    return {
        "image": img,
        "mask" : mask,
        "meta": meta,
        "prognosis": prognosis,
        "death": death
    }

def process_sample(
    img_file_name, meta, prognosis, death, config, augment=False
):
    img_path = tf.strings.join(
        [config.preprocessed_image_base_path, img_file_name], separator=os.path.sep)
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if augment:
        img = tf.image.random_jpeg_quality(img, 90, 100)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.clip_by_value(img, 0, 1)

    mask_path = tf.strings.join(
        [config.segmentation_base_path, img_file_name], separator=os.path.sep)
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    img = tf.image.resize(img, [config.img_height, config.img_width])
    mask = tf.image.resize(mask, [config.img_height, config.img_width])


    return {
        "image": img,
        "mask": mask,
        "prognosis": prognosis,
        "death": death,
        "meta": meta
    }

def get_dataset(table_path):
    df = pd.read_csv(table_path)

    death = df.pop("Death").to_numpy().flatten()
    prognosis = df.pop("Prognosis").to_numpy().flatten()
    prognosis = [0. if prog=='MILD' else 1. for prog in prognosis]
    image = df.pop("ImageFile").to_numpy().flatten()

    meta = df.to_numpy()

    return (
        np.array(image),
        np.array(meta),
        np.array(prognosis),
        np.array(death),
    )

def generate_data(config):
    train_image, train_meta, train_prognosis, train_death = get_dataset(config.train_table)
    valid_image, valid_meta, valid_prognosis, valid_death = get_dataset(config.valid_table)

    print("Number of train images found: ", len(train_image))
    print("Number of validation images found: ", len(valid_image))

    encode_single_sample_wrapped = partial(
        process_sample,
        config=config,
        augment=config.augment,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_image, train_meta, train_prognosis, train_death)
    )
    train_dataset = (
        train_dataset.map(
            encode_single_sample_wrapped,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            partial(
                process_data,
                augment=config.augment,
                img_width=config.img_width,
                img_height=config.img_height,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            partial(set_shapes, config=config),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .shuffle(buffer_size=5_000, reshuffle_each_iteration=True)
        .batch(config.batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    encode_single_sample_wrapped = partial(
        process_sample, config=config
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_image, valid_meta, valid_prognosis, valid_death)
    )
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample_wrapped,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            partial(
                process_data,
                augment=False,
                img_width=config.img_width,
                img_height=config.img_height,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            partial(set_shapes, config=config),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(config.batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    if config.visualize:
        _, ax = plt.subplots(2, 6, figsize=(12, 12))
        for batch in train_dataset.take(1):
            images = batch["image"]
            masks = batch["mask"]
            deaths = batch["death"]
            progs = batch["prognosis"]

            for i in range(min(6, config.batch_size)):
                img = (images[i] * 255.0).numpy().astype("uint8")
                mask = (masks[i] * 255.0).numpy().astype("uint8")
                ax[0, i].imshow(img, cmap="gray")
                ax[1, i].imshow(mask, cmap="gray")
                ax[0, i].set_title('Death: %d' % deaths.numpy()[i])
                ax[1, i].set_title('Prognosis: %d' % progs.numpy()[i])
                ax[0, i].axis("off")
                ax[1, i].axis("off")
                
        plt.savefig( f"{config.raw_output_base}/batch_sample.png", dpi=75)
        plt.close()

    return {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset
    }