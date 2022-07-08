import tensorflow as tf


class DenseBlock(tf.keras.Model):
    def __init__(self, bottleneck_num, growth_rate):
        super(DenseBlock, self).__init__(name="")
        # 根据conv_num新建卷积层对
        self.Bottleneck_list = []
        for i in range(bottleneck_num):
            BN1 = tf.keras.layers.BatchNormalization()  # BN层
            ReLU1 = tf.keras.layers.ReLU()              # ReLU层
            Conv1 = tf.keras.layers.Conv2D(growth_rate * 4, (1, 1))  # 1*1卷积核不加padding输入输出大小保持一致
            BN3 = tf.keras.layers.BatchNormalization()  # BN层
            ReLU3 = tf.keras.layers.ReLU()  # ReLU层
            Conv3 = tf.keras.layers.Conv2D(growth_rate, (3, 3), padding="same")
            self.Bottleneck_list.append([BN1, ReLU1, Conv1, BN3, ReLU3, Conv3])

    def call(self, input_tensor, training=False):  # 设置为非训练模式(推理模式)
        x = None
        output = None
        is_input = True
        for layers in self.Bottleneck_list:
            if is_input:
                x = layers[0](input_tensor)     # BN1
                x = layers[1](x)                # ReLU1
                x = layers[2](x)                # Conv1
                x = layers[3](x)                # BN3
                x = layers[4](x)                # ReLU3
                x = layers[5](x)                # Conv3
                output = tf.concat([x, input_tensor], axis=-1)  # concat
                is_input = not is_input
                continue
            x = layers[0](output)   # BN1
            x = layers[1](x)        # ReLU1
            x = layers[2](x)        # Conv1
            x = layers[3](x)        # BN3
            x = layers[4](x)        # ReLU3
            x = layers[5](x)        # Conv3
            output = tf.concat([x, output], axis=-1)
        return output


class Transition(tf.keras.Model):
    def __init__(self, out_channel):        # out_channel一般是输入通道数的0.5
        super(Transition, self).__init__(name="")

        self.Conv = tf.keras.layers.Conv2D(out_channel, (1, 1))
        self.BN = tf.keras.layers.BatchNormalization()
        self.Pool = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, input_tensor, training=False):
        x = self.BN(input_tensor)
        x = self.Conv(x)
        x = self.Pool(x)
        return x

class Classification(tf.keras.Model):
    def __init__(self, class_num):      # class_num输出的分类数量
        super(Classification, self).__init__(name="")
        self.global_avgpool = tf.keras.layers.GlobalAvgPool2D()
        self.Dense = tf.keras.layers.Dense(class_num)
        self.Softmax = tf.keras.layers.Softmax()

    def call(self, input_tensor, training=False):
        x = self.global_avgpool(input_tensor)
        x = self.Dense(x)
        x = self.Softmax(x)
        return x
