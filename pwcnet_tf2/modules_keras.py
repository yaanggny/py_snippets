import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='same', dilation_rate=1, name=None):
        super(Conv2D, self).__init__(name=name)

        self.out = layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       kernel_initializer='he_normal',
                                       dilation_rate=dilation_rate,
                                       activation=layers.LeakyReLU(0.1))

    def call(self, inputs):
        x = self.out(inputs)

        return x


class DeConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, name=None):
        super(DeConv2D, self).__init__(name=name)

        self.out = layers.Conv2DTranspose(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding='same',
                                                name=name)

    def call(self, inputs):
        x = self.out(inputs)

        return x


def FeaturePyramidExtractor_(input_shape=(438, 1024, 3), pyr_lvls=6):
    """
    方便调节层数
    """

    x = keras.Input(shape=input_shape, name='input')
    num_filters = [16, 32, 64, 96, 128, 196]
    pyr = []
    for lvl in range(0, pyr_lvls):
        f = num_filters[lvl]
        conv = layers.Conv2D(f, 3, 2, padding='same', kernel_initializer='he_normal',
                                   dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1),
                                   name=f'conv{lvl}a')(x)
        conv = layers.Conv2D(f, 3, 1, padding='same', kernel_initializer='he_normal',
                                   dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1),
                                   name=f'conv{lvl}aa')(conv)
        conv = layers.Conv2D(f, 3, 1, padding='same', kernel_initializer='he_normal',
                                   dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1),
                                   name=f'conv{lvl}b')(conv)
        pyr.append(conv)

    model = keras.Model(inputs=x, outputs=pyr)

    return model

def FeatureExtractorBase(input_shape=(438, 1024, 3), f=32, pyr_lvl=0):
    """
    方便调节层数
    """

    x = keras.Input(shape=input_shape, name='input')
    conv = layers.Conv2D(f, 3, 2, padding='same', kernel_initializer='he_normal',
                               dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1),
                               name=f'conv{pyr_lvl}a')(x)
    conv = layers.Conv2D(f, 3, 1, padding='same', kernel_initializer='he_normal',
                               dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1),
                               name=f'conv{pyr_lvl}aa')(conv)
    conv = layers.Conv2D(f, 3, 1, padding='same', kernel_initializer='he_normal',
                               dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1),
                               name=f'conv{pyr_lvl}b')(conv)
    model = keras.Model(inputs=x, outputs=conv)

    return model


def FeaturePyramidExtractor_2(input_shape=(2, 438, 1024, 3), pyr_lvls=6):
    """
    方便调节层数
    """

    x = keras.Input(shape=input_shape, name='input')
    x1, x2 = x[0], x[1]
    num_filters = [16, 32, 64, 96, 128, 196]

    pyr = []
    for lvl in range(0, pyr_lvls):
        f = num_filters[lvl]
        fe_base = FeatureExtractorBase(input_shape[1:], f, lvl)
        x1 = fe_base(x1)
        x2 = fe_base(x2)
        pyr.append((x1, x2))

    model = keras.Model(inputs=x, outputs=pyr)
    return model


class FeaturePyramidExtractor(keras.Model):
    """
    brief: pyramid feature extractiotor
    num_chann = [16, 32, 64, 96, 128, 196]
    """

    def __init__(self, max_displacement=4):
        super(FeaturePyramidExtractor, self).__init__()
        conv1 = layers.Conv2D(16, 3, 2, padding='same', kernel_initializer='he_normal',
                                    dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1))
        conv1 = layers.Conv2D(16, 3, 1, padding='same', kernel_initializer='he_normal',
                                    dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1))
        conv1 = layers.Conv2D(16, 3, 1, padding='same', kernel_initializer='he_normal',
                                    dilation_rate=1, activation=layers.LeakyReLU(alpha=0.1))

        num_chann = [16, 32, 64, 96, 128, 196]
        for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
            for lvl in range(1, self.opts['pyr_lvls'] + 1):
                f = num_chann[lvl]
                x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name=f'conv{lvl}a', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}a') # default alpha is 0.2 for TF
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}aa', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}aa')
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}b', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1, name=f'{name}{lvl}')
                pyr.append(x)

        self.conv1a = Conv2D(16, kernel_size=3, strides=2, name='conv1a')
        self.conv1aa = Conv2D(16, kernel_size=3, strides=1, name='conv1aa')
        self.conv1b = Conv2D(16, kernel_size=3, strides=1, name='conv1b')
        self.conv2a = Conv2D(32, kernel_size=3, strides=2, name='conv2a')
        self.conv2aa = Conv2D(32, kernel_size=3, strides=1, name='conv2aa')
        self.conv2b = Conv2D(32, kernel_size=3, strides=1, name='conv2b')
        self.conv3a = Conv2D(64, kernel_size=3, strides=2, name='conv3a')
        self.conv3aa = Conv2D(64, kernel_size=3, strides=1, name='conv3aa')
        self.conv3b = Conv2D(64, kernel_size=3, strides=1, name='conv3b')
        self.conv4a = Conv2D(96, kernel_size=3, strides=2, name='conv4a')
        self.conv4aa = Conv2D(96, kernel_size=3, strides=1, name='conv4aa')
        self.conv4b = Conv2D(96, kernel_size=3, strides=1, name='conv4b')
        self.conv5a = Conv2D(128, kernel_size=3, strides=2, name='conv5a')
        self.conv5aa = Conv2D(128, kernel_size=3, strides=1, name='conv5aa')
        self.conv5b = Conv2D(128, kernel_size=3, strides=1, name='conv5b')
        self.conv6aa = Conv2D(196, kernel_size=3, strides=2, name='conv6aa')
        self.conv6a = Conv2D(196, kernel_size=3, strides=1, name='conv6a')
        self.conv6b = Conv2D(196, kernel_size=3, strides=1, name='conv6b')

def cost_volume(c1, warp, search_range, level=0, name='cost_volume'):
    """
    cost volume
    """

    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return cost_vol

class FlowPredictor(layers.Layer):
    """
    flow predict
    """

    def __init__(self, name='predict_flow'):
        super(FlowPredictor, self).__init__()
        self.conv = layers.Conv2D(2, 3, 1, padding='same', name=name)

    def call(self, inputs):
        return self.conv(inputs)

