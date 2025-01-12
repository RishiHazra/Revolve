import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    BatchNormalization,
    Input,
    Lambda,
)
from tensorflow.keras.layers import GaussianNoise


# import tensorflow_addons as tfa


class DDDQN(tf.keras.Model):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.activation = "swish"
        self.dropout = 0.4
        self.initializer = HeNormal(seed=None)

    def create_model(self):
        input_ = Input(shape=(100, 256, 12))
        activation = self.activation
        dropout = self.dropout
        initializer = self.initializer

        optimizer = tf.keras.optimizers.Adam(lr=0.0005, clipnorm=1.5)

        x = Lambda(lambda x: x / 255.0)(input_)
        x = GaussianNoise(stddev=0.125)(x)
        x = Conv2D(16, (5, 5), padding="same", activation=activation, name="Conv1")(x)
        x = BatchNormalization(name="bn1")(x)

        x = Conv2D(32, (3, 3), padding="same", activation=activation, name="Conv2")(x)
        x = BatchNormalization(name="bn2")(x)

        x = Conv2D(
            64,
            (3, 3),
            padding="same",
            strides=(2, 2),
            activation=activation,
            name="Conv3",
        )(x)
        x = BatchNormalization(name="bn3")(x)

        x = Conv2D(
            128,
            (3, 3),
            padding="same",
            strides=(2, 2),
            activation=activation,
            name="Conv4",
        )(x)
        x = BatchNormalization(name="bn4")(x)

        x = Conv2D(
            256,
            (3, 3),
            padding="same",
            strides=(2, 2),
            activation=activation,
            name="Conv5",
        )(x)
        x = BatchNormalization(name="bn5")(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = Flatten()(x)
        x = layers.Dense(256, activation=activation, name="dense1")(x)
        x = BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

        input2 = Input(shape=(6))  # 2 euleor angles vx,vy, absolute speed
        input2_processed = Dense(
            256, kernel_initializer=initializer, activation=activation
        )(input2)
        x2 = BatchNormalization()(input2_processed)
        x2 = layers.Dropout(dropout)(x2)

        x = layers.Concatenate()([x, x2])

        value = layers.Dense(
            128, kernel_initializer=initializer, activation=activation
        )(x)
        value = tf.keras.layers.LayerNormalization()(value)
        value = layers.Dropout(dropout)(value)
        value = layers.Dense(1, kernel_initializer=initializer, activation="linear")(
            value
        )

        # Advantage stream
        advantage = layers.Dense(
            128, kernel_initializer=initializer, activation=activation, name="name1"
        )(x)
        advantage = tf.keras.layers.LayerNormalization()(advantage)
        advantage = layers.Dropout(dropout, name="name3")(advantage)
        advantage = layers.Dense(
            66, kernel_initializer=initializer, activation="linear", name="name4"
        )(advantage)

        # Combine value and advantage to get Q-values
        outputs = layers.Add()(
            [
                value,
                layers.Subtract()(
                    [advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)]
                ),
            ]
        )  # add layer norm before q values

        model = tf.keras.Model([input_, input2], outputs)
        # model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=0.25), metrics=['accuracy'])  # can be
        # tested
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)

        return model
