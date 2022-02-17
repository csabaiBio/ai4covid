import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from albumentations import (
    Compose,
    GaussianBlur,
    RandomBrightnessContrast,
    RandomCrop,
    Resize,
    Rotate,
)


def aug_fn(image, mask, augment, width, height):
    data = {"image": image, "mask": mask}
    if augment:
        transforms = Compose(
            [
                RandomCrop(
                    int(width - 0.15 * width), int(height - 0.15 * height), p=0.5
                ),
                RandomBrightnessContrast(p=0.2),
                Rotate(limit=10),
                GaussianBlur(blur_limit=1.2),
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
    return aug_data["image"], aug_data["mask"]


def augment_data(sample, augment, img_width, img_height):
    image = sample["image"]
    mask = sample["mask"]
    aug_data = tf.numpy_function(
        func=aug_fn,
        inp=[image, mask, augment, img_width, img_height],
        Tout=[tf.float32, tf.float32],
    )
    sample["image"] = aug_data[0]
    sample["mask"] = aug_data[1]
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
    prognosis.set_shape((2))
    death.set_shape((2))

    return {
        "image": img,
        "mask": mask,
        "meta": meta,
        "prognosis": prognosis,
        "death": death,
    }


def process_sample(img_file_name, meta, prognosis, death, config, augment=False):
    img_path = tf.strings.join(
        [config.preprocessed_image_base_path, img_file_name], separator=os.path.sep
    )
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask_path = tf.strings.join(
        [config.segmentation_base_path, img_file_name], separator=os.path.sep
    )
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return {
        "image": img,
        "mask": mask,
        "prognosis": prognosis,
        "death": death,
        "meta": meta,
    }


def get_dataset(table_path):
    df = pd.read_csv(table_path)

    death = df.pop("Death").to_numpy().flatten().astype(int)
    prognosis = df.pop("Prognosis").to_numpy().flatten()
    prognosis = np.array([0 if prog == "MILD" else 1 for prog in prognosis]).astype(int)
    image = df.pop("ImageFile").to_numpy().flatten()

    prognosis = tf.keras.utils.to_categorical(prognosis, num_classes=2)
    death = tf.keras.utils.to_categorical(death, num_classes=2)

    meta = df.to_numpy()

    return (
        image,
        meta,
        prognosis,
        death,
    )


def generate_data(config, fold=None):
    if fold is not None and config.cross_val_train:
        train_image, train_meta, train_prognosis, train_death = get_dataset(
            config.cv_train_table + f"cv{fold + 1}.csv"
        )
        valid_image, valid_meta, valid_prognosis, valid_death = get_dataset(
            config.cv_valid_table + f"cv{fold + 1}.csv"
        )
    else:
        train_image, train_meta, train_prognosis, train_death = get_dataset(
            config.train_table
        )
        valid_image, valid_meta, valid_prognosis, valid_death = get_dataset(
            config.valid_table
        )

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
                augment_data,
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
        .repeat()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    encode_single_sample_wrapped = partial(process_sample, config=config)

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
                augment_data,
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
        _, ax = plt.subplots(2, 6, figsize=(10, 5))
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
                ax[0, i].set_title("Death: %d" % np.argmax(deaths.numpy()[i]))
                ax[1, i].set_title("Prognosis: %d" % np.argmax(progs.numpy()[i]))
                ax[0, i].axis("off")
                ax[1, i].axis("off")

        plt.savefig(f"{config.raw_output_base}/batch_sample.png", dpi=75)
        plt.close()

    return {"train_dataset": train_dataset, "validation_dataset": validation_dataset}


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="conf", config_name="train")
    def run(config):
        datasets = generate_data(config, 0)
        for sample in datasets["train_dataset"].take(2):
            print(sample["image"].numpy().shape)
            print(sample["mask"].numpy().shape)
            print(sample["death"].numpy().shape)
            print(sample["prognosis"].numpy().shape)
            print(sample["meta"].numpy().shape)
            print("*" * 50)

    run()
