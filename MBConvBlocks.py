import tensorflow as tf
import tf_keras as keras
import math
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose,GlobalAveragePooling2D,DepthwiseConv2D

from tensorflow.keras import regularizers

NUM_CLASSES = 5

################################ THE OPTIMAL WATT-EFFNET (3-6) CONFIGURATION CODE ###################################################################

def swish(x):
    return x * tf.keras.ops.sigmoid(x)


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = GlobalAveragePooling2D()
        self.reduce_conv = Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1, kernel_regularizer = regularizers.L2(1e-4), padding="same")

        self.expand_conv = Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1, kernel_regularizer = regularizers.L2(1e-4),
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.keras.ops.expand_dims(branch, axis=1)
        branch = tf.keras.ops.expand_dims(branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.keras.ops.sigmoid(branch)
        output = inputs * branch
        return output

    def from_config(cls, config):
        return cls(**config)


class MBConv(Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = Conv2D(filters=in_channels * expansion_factor,         
                                            kernel_size=(1, 1),
                                            strides=1, kernel_regularizer = regularizers.L2(1e-4),
                                            padding="same")
        self.bn1 = BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")

        self.se = SEBlock(input_channels=in_channels * expansion_factor)      
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1, kernel_regularizer = regularizers.L2(1e-4),
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = Dropout(rate=drop_connect_rate)



    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)

        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = Add()([x, inputs])
        return x

    def from_config(cls, config):
        return cls(**config)


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = keras.Sequential()

    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))

    return block
