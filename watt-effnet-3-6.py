class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.5):
        super(EfficientNet, self).__init__()

        self.conv1 = Conv2D(filters=round_filters(32, width_coefficient),   # Ideal : 32
                                            kernel_size=(3, 3),
                                            strides=2, kernel_regularizer = regularizers.L2(1e-4),
                                            padding="same")
        self.bn1 = BatchNormalization()



        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),   # 32: Ideal 32
                                         out_channels=round_filters(16, width_coefficient),  # 16: Ideal 16
                                         layers=round_repeats(1, depth_coefficient),  #1
                                         stride=1,  # 1, 1, 3.
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)   # CHANGE EXPANSION FACTOR HERE!

        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),     # 16: Ideal 16
                                         out_channels=round_filters(24, width_coefficient),    # 24: Ideal 24
                                         layers=round_repeats(2, depth_coefficient),  #2
                                         stride=2,   # 2,6,3
                                         expansion_factor = 6, k=3, drop_connect_rate=drop_connect_rate)   # CHANGE EXPANSION FACTOR HERE!

        self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),     # 24: Ideal 24
                                         out_channels=round_filters(40, width_coefficient),    # 40: Ideal 40
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,   #2,6,5
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)     # CHANGE EXPANSION FACTOR HERE!  k = 5

        self.conv2 = Conv2D(filters=round_filters(32, width_coefficient),   # Ideal: 32
                                            kernel_size=(3, 3),
                                            strides=2, kernel_regularizer = regularizers.L2(1e-4),
                                            padding="same")


        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(filters=round_filters(64, width_coefficient),   #Ideal: 64
                                            kernel_size=(3, 3),
                                            strides=2, kernel_regularizer = regularizers.L2(1e-4),
                                            padding="same")


        self.bn3 = BatchNormalization()

        self.conv4 = Conv2D(filters=round_filters(32, width_coefficient),   # Ideal: 32
                                            kernel_size=(3, 3),
                                            strides=2, kernel_regularizer = regularizers.L2(1e-4),
                                            padding="same")

        self.bn4 = BatchNormalization()

        self.pool = GlobalAveragePooling2D()
        self.dropout = Dropout(rate=dropout_rate)
        self.fc = Dense(units=5,activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.block1(x)
       #xc1 = cbam_block(x,4)
       #x = Add()([x,xc1])
        x = self.block2(x)
       #xc2 = cbam_block(x,4)
       #x = Add()([x,xc2])
        x = self.block3(x)
       #xc3 = cbam_block(x,4)
       #x = Add()([x,xc3])


        x = self.conv2(inputs)
        x = self.bn2(x, training=training)
        x = swish(x)
        x = Add()([x,x])
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = swish(x)
        x = Add()([x,x])
        x = self.conv4(x)
        x = self.bn4(x, training=training)
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
    net.call(Input(shape=(resolution, resolution, 3)))

    return net

Watteffnet36 =  get_efficient_net(1.0, 1.0, 224, 0.1)
Watteffnet36.summary()  
