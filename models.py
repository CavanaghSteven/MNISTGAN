import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
from params import Params


def activation(discrim):
    if discrim:
        return LeakyReLU()
    else:
        return ReLU()


def normalise(discrim):
    if discrim:
        return LayerNormalization()
    else:
        return BatchNormalization()


def res_block(num_filters, filter_size, strides, input_layer, discrim=False):
    shortcut_connection = input_layer

    conv = Conv2D(num_filters, filter_size, strides=strides, padding='same', kernel_initializer='he_normal')(
        input_layer)
    conv = normalise(discrim)(conv)
    conv = activation(discrim)(conv)

    # Check if shortcut connection need a conv layer
    filters_equal = np.shape(conv)[-1] != np.shape(shortcut_connection)[-1]
    # print('equal ?', filters_equal)
    if strides != 1 or filters_equal:
        shortcut_connection = Conv2D(
            num_filters, 1, strides=strides, padding='same', kernel_initializer='he_normal'
        )(shortcut_connection)

    res = Add()([shortcut_connection, conv])
    res = activation(discrim)(res)
    return res


def get_discriminator(params: Params):
    in_layer = Input(shape=params.img_dim)

    conv_1 = res_block(16, 6, 2, in_layer, discrim=True)
    conv_2 = res_block(32, 6, 1, conv_1, discrim=True)
    conv_3 = res_block(64, 6, 2, conv_2, discrim=True)
    conv_4 = res_block(128, 6, 1, conv_3, discrim=True)

    flatten = Flatten()(conv_4)

    dense_1 = Dense(128, activation='relu')(flatten)
    dense_1 = Dropout(0.1)(dense_1)

    output = Dense(1, activation='linear')(dense_1)

    model = Model(inputs=in_layer, outputs=output, name='discriminator')

    return model


def get_generator(params: Params):
    in_layer = Input(shape=(params.latent_dim,))

    dense_1 = Dense(7 * 7 * 128, activation=None, kernel_initializer='he_normal')(in_layer)
    # dense_1 = LeakyReLU(alpha=0.5)(dense_1)

    reshaped = Reshape(target_shape=(7, 7, 128))(dense_1)

    conv_1 = UpSampling2D(interpolation='bilinear')(reshaped)
    conv_1 = res_block(64, 6, 1, conv_1)

    conv_2 = UpSampling2D(interpolation='bilinear')(conv_1)
    conv_2 = res_block(64, 6, 1, conv_2)

    conv_3 = res_block(64, 6, 1, conv_2)
    conv_4 = res_block(64, 6, 1, conv_3)

    out_layer = Conv2D(1, kernel_size=6, strides=1, activation='tanh', padding='same')(
        conv_4)

    model = Model(inputs=in_layer, outputs=out_layer, name='Generator')
    # model.compile(loss='mse', optimizer=Adam())

    return model


if __name__ == '__main__':
    params = Params()
    gen = get_generator(params)
    dis = get_discriminator(params)

    print('generator', gen.count_params())
    print('discrim', dis.count_params())

    # print('generator', gen.summary())
    # print('discrim', dis.summary())
