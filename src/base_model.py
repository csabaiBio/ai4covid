import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
)

tf.random.set_seed(137)

class BinaryLossLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, y_true, y_pred, rate):
        loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=True
        )
        self.add_loss(rate * tf.reduce_mean(loss))
        return y_pred

def build_model(config):
    input_img = Input(
        shape=(config.img_width, config.img_height, config.channels),
        name="image",
        dtype="float32",
    )
    input_mask = Input(
        shape=(config.img_width, config.img_height, config.channels),
        name="mask",
        dtype="float32",
    )

    input_raw = Concatenate(axis=-1)([input_img, input_mask])

    input_meta = Input(
        shape=(config.n_feature_cols),
        name="meta",
        dtype="float32"
    )

    death = Input(name="death", shape=(None,), dtype="float32")
    prognosis = Input(name="prognosis", shape=(None,), dtype="float32")

    ## CONV HEAD
    _x = Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1_1",
    )(input_raw)
    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1_2",
    )(_x)
    _x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1_skip",
    )(_x)
    x = Concatenate()([x, _x])
    x = BatchNormalization(name="BN_1")(x)

    _x = MaxPooling2D((2, 2), name="pool1")(x)  # (None, 128, 16, 32)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2_1",
    )(_x)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2_2",
    )(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2_3",
    )(x)

    _x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2_skip",
    )(_x)

    x = Concatenate()([x, _x])
    x = BatchNormalization(name="BN_2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    x = Conv2D(
        256,
        (2, 2),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3_1",
    )(x)
    x = Conv2D(
        256,
        (2, 2),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3_2",
    )(x)
    x = BatchNormalization(name="BN_3")(x)
    x = MaxPooling2D((1, 2), name="pool3")(x)
    x = Conv2D(
        256,
        (2, 1),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv4_1",
    )(x)
    x = Conv2D(
        512,
        (2, 1),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv4_2",
    )(x)

    x = BatchNormalization(name="BN_4")(x)
    image_out = Flatten(name="image_flatten")(x)

    ## META HEAD
    meta = Dense(64, activation='relu')(input_meta)
    meta = BatchNormalization(name="meta_bn")(meta)
    meta = Dense(128, activation='relu')(meta)
    meta = BatchNormalization(name="meta_bn")(meta)
    meta = Dense(256, activation='relu')(meta)
    meta = BatchNormalization(name="meta_bn")(meta)
    meta_out = Flatten(name="meta_flatten")(meta)

    ## COMBINED HEAD
    combined = Concatenate(axis=-1)([image_out, meta_out])
    out = Dense(256, activation='relu')(combined)
    out = BatchNormalization(name="meta_bn")(out)
    out = Dropout(.2)(out)
    out = Dense(32, activation='relu')(out)
    out = BatchNormalization(name="meta_bn")(out)
    out = Dense(2, activation=None)(out)

    p, d = tf.unstack(out, axis=-1)

    prognosis_out, death_out = BinaryLossLayer(name="prognosis_out")(prognosis, ), BinaryLossLayer(name="death_out")(death, ) 

    model = keras.models.Model(
        inputs=[input_img, input_mask, input_meta, prognosis, death],
        outputs=[prognosis, death],
        name="model_v1",
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate, decay_steps=500, decay_rate=0.96, staircase=False
    )

    opt = keras.optimizers.Adam(lr_schedule)
    model.compile(optimizer=opt)
    return model