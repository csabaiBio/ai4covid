from pydoc import locate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Softmax,
)

from src.balanced_accuracy import BACC

tf.random.set_seed(137)


backbone_dict = {
    "EfficientNetB0": tf.keras.applications.efficientnet.EfficientNetB0,
    "EfficientNetB1": tf.keras.applications.efficientnet.EfficientNetB1,
    "ResNet50": tf.keras.applications.resnet.ResNet50,
    "VGG16": tf.keras.applications.vgg16.VGG16,
    "VGG19": tf.keras.applications.vgg19.VGG19,
    "InceptionV3": tf.keras.applications.inception_v3.InceptionV3,
}


class BinaryEndpointLayer(layers.Layer):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        if name == "prognosis_binary_loss":
            self.binary_metrics = [
                BACC(name="balanced_accuracy"),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.BinaryCrossentropy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(name="sensitivity"),
                tf.keras.metrics.PrecisionAtRecall(
                    0.5, name="precision_at_50_pct_recall"
                ),
                tf.keras.metrics.PrecisionAtRecall(
                    0.75, name="precision_at_75_pct_recall"
                ),
                tf.keras.metrics.PrecisionAtRecall(
                    0.95, name="precision_at_95_pct_recall"
                ),
                tf.keras.metrics.SpecificityAtSensitivity(
                    0.5, name="specificity_at_50_pct_recall"
                ),
                tf.keras.metrics.SpecificityAtSensitivity(
                    0.75, name="specificity_at_75_pct_recall"
                ),
                tf.keras.metrics.SpecificityAtSensitivity(
                    0.95, name="specificity_at_95_pct_recall"
                ),
            ]
            self.rate = 1.0
        else:
            self.binary_metrics = [
                BACC(name="death_balanced_accuracy"),
                tf.keras.metrics.BinaryAccuracy(name="death_binary_accuracy"),
                tf.keras.metrics.BinaryCrossentropy(name="death_binary_crossentropy"),
                tf.keras.metrics.Precision(name="death_precision"),
            ]
            self.rate = config.death_rate

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        self.add_loss(self.rate * loss)
        for metric in self.binary_metrics:
            self.add_metric(metric(y_true, y_pred))
        return y_pred


def build_model(config):
    input_img = Input(
        shape=(config.img_size, config.img_size, 1),
        name="image",
        dtype="float32",
    )
    input_mask = Input(
        shape=(config.img_size, config.img_size, 1),
        name="mask",
        dtype="float32",
    )
    input_fourier = Input(
        shape=(config.img_size, config.img_size, 1),
        name="fourier",
        dtype="float32",
    )

    input_raw = Concatenate(axis=-1, name="concat_inputs")([input_img, input_mask])

    n_feature_cols = len(config.datasets[config.dataset_identifier].feature_cols)
    input_meta = Input(shape=(n_feature_cols), name="meta", dtype="float32")

    input_brixia = Input(shape=(6), name="brixia", dtype="float32")

    meta = Concatenate(axis=-1, name="meta_and_brixia")([input_meta, input_brixia])

    death = Input(name="death", shape=(2), dtype="int32")
    prognosis = Input(name="prognosis", shape=(2), dtype="int32")

    # IMAGING HEAD
    backbone = backbone_dict[config.backbone](
        include_top=False,
        weights=None,
        input_shape=(config.img_size, config.img_size, 2),
        pooling="avg",
    )
    backbone.trainable = True
    image_out = backbone(input_raw)

    ## FOURIER HEAD
    fourier = Conv2D(16, (7, 7), strides=(2, 2), activation="relu")(input_fourier)
    fourier = BatchNormalization(name="fourier_bn1")(fourier)
    fourier = Conv2D(32, (3, 3), strides=(2, 2), activation="relu")(fourier)
    fourier = BatchNormalization(name="fourier_bn2")(fourier)
    fourier = Conv2D(64, (3, 3), strides=(2, 2), activation="relu")(fourier)
    fourier = BatchNormalization(name="fourier_bn3")(fourier)
    fourier = Conv2D(128, (3, 3), strides=(2, 2), activation="relu")(fourier)
    fourier = BatchNormalization(name="fourier_bn4")(fourier)
    fourier = Conv2D(256, (3, 3), strides=(2, 2), activation="relu")(fourier)
    fourier = BatchNormalization(name="fourier_bn5")(fourier)
    fourier_out = GlobalAveragePooling2D(name="fourier_avg_pool")(fourier)

    ## META HEAD
    meta = Dense(128, activation="relu")(meta)
    meta = BatchNormalization(name="meta_bn1")(meta)
    meta = Dense(256, activation="relu")(meta)
    meta = BatchNormalization(name="meta_bn2")(meta)
    meta = Dense(512, activation="relu")(meta)
    meta = BatchNormalization(name="meta_bn3")(meta)
    meta = Dense(1024, activation="relu")(meta)
    meta_out = BatchNormalization(name="meta_bn4")(meta)

    ## COMBINED HEAD
    combined = Concatenate(axis=-1, name="concat_heads")(
        [image_out, fourier_out, meta_out]
    )
    out = Dense(512, activation="relu")(combined)
    out = BatchNormalization(name="head_bn1")(out)
    out = Dropout(0.2)(out)
    out = Dense(256, activation="relu")(out)
    out = BatchNormalization(name="head_bn2")(out)
    out = Dropout(0.2)(out)
    out = Dense(config.last_dense_size, activation="relu")(out)
    out = BatchNormalization(name="head_bn3")(out)
    out = Dense(4, activation="linear", name="unnormalized_output")(out)

    p, d = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1), name="outputs")(
        out
    )
    p = Softmax(name="prognosis_out")(p)
    d = Softmax(name="death_out")(d)

    prognosis_out = BinaryEndpointLayer(name="prognosis_binary_loss", config=config)(
        prognosis, p
    )
    death_out = BinaryEndpointLayer(name="death_binary_loss", config=config)(death, d)

    model = keras.models.Model(
        inputs=[
            input_img,
            input_mask,
            input_fourier,
            input_meta,
            input_brixia,
            prognosis,
            death,
        ],
        outputs=[prognosis_out, death_out],
        name="model_v1",
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate,
        decay_steps=3 * config.steps_per_epoch,
        decay_rate=0.15,
        staircase=False,
    )

    opt = keras.optimizers.Adam(lr_schedule)
    model.compile(optimizer=opt)

    return model
