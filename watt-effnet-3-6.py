import tensorflow as tf
import math
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose,GlobalAveragePooling2D,DepthwiseConv2D

from tensorflow.keras import regularizers

NUM_CLASSES = 5

################################ THE OPTIMAL WATT-EFFNET (3-6) CONFIGURATION CODE ###################################################################

def swish(x):
    return x * tf.nn.sigmoid(x)


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
                                                  strides=1, kernel_regularizer = regularizers.L2(1e-6), padding="same")

        self.expand_conv = Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1, kernel_regularizer = regularizers.L2(1e-6),
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
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
        self.conv1 = Conv2D(filters=in_channels * expansion_factor,         # WIDTH ????
                                            kernel_size=(1, 1),
                                            strides=1, kernel_regularizer = regularizers.L2(1e-6),
                                            padding="same")
        self.bn1 = BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dwconv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")
        self.bn22 = BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)       # WIDTH ????
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1, kernel_regularizer = regularizers.L2(1e-6),
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = Dropout(rate=drop_connect_rate)



    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.dwconv2(x)
        x = self.bn22(x, training=training)

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
    block = tf.keras.Sequential()
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


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.5):
        super(EfficientNet, self).__init__()

        self.conv1 = Conv2D(filters=round_filters(32, width_coefficient),   #32
                                            kernel_size=(3, 3),
                                            strides=2, kernel_regularizer = regularizers.L2(1e-6),
                                            padding="same")
        self.bn1 = BatchNormalization()
        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),  #1
                                         stride=1,  # 1, 1, 3.
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)   # CHANGE EXPANSION FACTOR HERE!
        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),  #2
                                         stride=2,   # 2,6,3
                                         expansion_factor = 36, k=3, drop_connect_rate=drop_connect_rate)   # CHANGE EXPANSION FACTOR HERE!
        #self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                         #out_channels=round_filters(40, width_coefficient),
                                         #layers=round_repeats(2, depth_coefficient),
                                         #stride=2,   #2,6,5
                                         #expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)

        self.pool = GlobalAveragePooling2D()
        self.dropout = Dropout(rate=dropout_rate)
        self.fc = Dense(units=5,activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.block1(x)
        #x = cbam_block(x,4)
        x = self.block2(x)
        #x = cbam_block(x,4)
        #x = self.block3(x)
        x = swish(x)
        x = Add()([x,x])
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        return x

    def from_config(cls, config):
        return cls(**config)

def get_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate):
    net = EfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate)
    net.build(input_shape=(None, resolution, resolution, 3))
    net.call(Input(shape=(resolution, resolution, 3)))

    return net

def efficient_net_b0():
    return get_efficient_net(1.0, 1.0, 224, 0.2) 

Watteffnet36 =  get_efficient_net(1.0, 1.0, 224, 0.1)    
#model = Model(Input(224,224,3), efficient_net_b0)
Watteffnet36.summary()
