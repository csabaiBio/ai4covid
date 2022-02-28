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
    Lambda,
    Softmax,
)

from src.balanced_accuracy import BACC

tf.random.set_seed(137)


class BinaryEndpointLayer(layers.Layer):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        if name == "prognosis_binary_loss":
            self.binary_metrics = [
                BACC(name="balanced_accuracy"),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.BinaryCrossentropy(),
            ]
            self.rate = 1.0
            self.pos_weight = tf.constant(1.0 / (568.0 / 538.0))
        else:
            self.binary_metrics = [
                BACC(name="death_balanced_accuracy"),
                tf.keras.metrics.BinaryAccuracy(name="death_binary_accuracy"),
                tf.keras.metrics.BinaryCrossentropy(name="death_binary_crossentropy"),
            ]
            self.rate = config.death_rate
            self.pos_weight = tf.constant(1.0 / (193.0 / 910.0))

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(tf.argmax(y_true, axis=-1), (-1, 1)), tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            y_true, logits=y_pred, pos_weight=self.pos_weight
        )
        self.add_loss(self.rate * loss)
        for metric in self.binary_metrics:
            self.add_metric(metric(y_true, tf.nn.sigmoid(y_pred)))
        return tf.nn.sigmoid(y_pred)


backbone_dict = {
    "EfficientNetB0": tf.keras.applications.efficientnet.EfficientNetB0,
    "EfficientNetB1": tf.keras.applications.efficientnet.EfficientNetB1,
    "ResNet50": tf.keras.applications.resnet.ResNet50,
    "VGG16": tf.keras.applications.vgg16.VGG16,
    "VGG19": tf.keras.applications.vgg19.VGG19,
    "InceptionV3": tf.keras.applications.inception_v3.InceptionV3,
}


def get_cnn_model(config):
    base_model = backbone_dict[config.backbone](
        input_shape=(config.img_size, config.img_size, 2),
        include_top=False,
        weights=None,
    )
    base_model.trainable = True
    base_model_out = base_model.layers[-1].output
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = 1
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=1, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(dense_dim, activation="relu")

    def call(self, inputs, training):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = Conv2D(embedding_dim, (1, 1), padding="same", activation="relu")

    def call(self, x):
        x = self.fc(x)
        return x


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, n_hidden):
        super(BahdanauAttention, self).__init__()
        self.W1s = [tf.keras.layers.Dense(units) for _ in range(n_hidden)]
        self.W2s = [tf.keras.layers.Dense(units) for _ in range(n_hidden)]
        self.Vs = [tf.keras.layers.Dense(1) for _ in range(n_hidden)]

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, n_hidden, hidden_size)
        hidden_with_time_axis = tf.unstack(hidden, axis=1)

        all_context_vectors = []
        all_attention_weights = []

        for ind, hidden in enumerate(hidden_with_time_axis):
            hidden = tf.expand_dims(hidden, 1)
            # attention_hidden_layer shape == (batch_size, 64, 1, units)
            attention_hidden_layer = tf.nn.tanh(
                self.W1s[ind](features) + self.W2s[ind](hidden)
            )

            # score shape == (batch_size, 64, 1, 1)
            # This gives you an unnormalized score for each image feature.
            score = self.Vs[ind](attention_hidden_layer)

            # attention_weights shape == (batch_size, 64, 1, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # attention_weights shape == (batch_size, 64, 1, 1)
            # features shape == (batch_size, 64, 1, embedding_dim)
            # context_vector shape ==  (batch_size, 64, 1, embedding_dim)
            context_vector = attention_weights * features
            # context_vector shape after sum == (batch_size, 1, embedding_dim) -> 1, 2048
            context_vector = tf.reduce_sum(context_vector, axis=1)

            all_context_vectors.append(context_vector)
            all_attention_weights.append(attention_weights)

        context_vector = tf.stack(all_context_vectors, axis=1)
        attention_weights = tf.stack(all_attention_weights, axis=2)

        return context_vector, attention_weights


def build_xplainable_model(config):
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

    input_raw = Concatenate(axis=-1, name="concat_inputs")([input_img, input_mask])

    n_feature_cols = len(config.datasets[config.dataset_identifier].feature_cols)

    input_meta = Input(shape=(n_feature_cols), name="meta", dtype="float32")
    meta = layers.Reshape(target_shape=(n_feature_cols, 1), name="meta_reshape")(
        input_meta
    )

    death = Input(name="death", shape=(2), dtype="int32")
    prognosis = Input(name="prognosis", shape=(2), dtype="int32")

    cnn_model = get_cnn_model(config)
    cnn_encoder = CNN_Encoder(config.cnn_encode_dim)
    meta_encoder = TransformerEncoderBlock(
        config.transformer_encode_dim, config.transformer_heads
    )
    attention = BahdanauAttention(config.bahdanau_dim, n_feature_cols)

    encoded_img = cnn_model(input_raw)
    encoded_img = cnn_encoder(encoded_img)
    encoded_meta = meta_encoder(meta)

    encoded_img = layers.Reshape(
        target_shape=(-1, config.cnn_encode_dim), name="reshape_encoded_image"
    )(encoded_img)

    context_vector, attention_weights = attention(encoded_img, encoded_meta)

    attention_weights = Lambda(lambda x: x, name="attention_weights")(attention_weights)

    ## PREDICTION HEAD
    out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.35))(context_vector)
    out = BatchNormalization(name="lstm_bn1")(out)
    out = Bidirectional(LSTM(128, return_sequences=False, dropout=0.35))(out)
    out = BatchNormalization(name="lstm_bn2")(out)
    out = Dense(64, activation="relu", name="dense_out_1")(out)
    out = Dense(2, activation="linear", name="dense_out_2")(out)

    p, d = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1), name="outputs")(
        out
    )

    p_out = Lambda(lambda x: tf.nn.sigmoid(x), name="prognosis_out")(p)

    prognosis_out = BinaryEndpointLayer(name="prognosis_binary_loss", config=config)(
        prognosis, p
    )
    death_out = BinaryEndpointLayer(name="death_binary_loss", config=config)(death, d)

    model = keras.models.Model(
        inputs=[input_img, input_mask, input_meta, prognosis, death],
        outputs=[prognosis_out, death_out, attention_weights, p_out],
        name="model_v1",
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate,
        decay_steps=3 * config.steps_per_epoch,
        decay_rate=0.96,
        staircase=False,
    )

    opt = keras.optimizers.Adam(lr_schedule)
    model.compile(optimizer=opt)

    return model


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="conf", config_name="train_xplain")
    def run(config):
        model = build_xplainable_model(config)
        model.summary()

    run()
