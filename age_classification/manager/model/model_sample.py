import tensorflow as tf

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import InputLayer
from keras.applications import vgg16
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization


def build_model_v0():
    models = tf.keras.Sequential([
        InputLayer(input_shape=(48, 48, 1)),
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(rate=0.5),
        Dense(1, activation='relu')
    ])

    optimizer_sgd = tf.keras.optimizers.SGD(momentum=0.9)
    return models.compile(optimizer=optimizer_sgd,
                          loss='mean_squared_error',
                          metrics=['mae'])


def vgg16_model_v0():
    losses = ['mse']
    learning_rate = 1e-4
    input_layer = Input((48, 48, 3), dtype=tf.uint8)
    model_layer = tf.cast(input_layer, tf.float32)
    model_vgg16 = vgg16.VGG16(include_top=False, weights='imagenet',
                              input_tensor=vgg16.preprocess_input(model_layer))
    model_layer = model_vgg16.get_layer("block5_conv3").output
    model_layer = Flatten()(model_layer)
    model_layer = Dense(1024, activation=LeakyReLU(alpha=0.3))(model_layer)
    model_layer = BatchNormalization()(model_layer)
    model_layer = Dropout(0.2)(model_layer)
    output_layer = Dense(1, name='age_output')(model_layer)

    model = tf.keras.Model(input_layer, [output_layer])
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=losses, metrics={'age_output': 'mean_absolute_error'})
    return model