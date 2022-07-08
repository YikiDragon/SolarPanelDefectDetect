import tensorflow as tf
import DenseNet


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__(name="DenseNetModel")
        # 输入[1,195,195,1]
        self.Conv = tf.keras.layers.Conv2D(16, (6, 6), strides=(3, 3))
        # [1,64,64,16]
        self.Pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        # [1,32,32,16]
        self.DenseBlock1 = DenseNet.DenseBlock(6, 12)  # 6组Conv-BN-ReLU 增长率12
        # [1,32,32,88]
        self.Transition1 = DenseNet.Transition(44)
        # [1,16,16,44]
        self.DenseBlock2 = DenseNet.DenseBlock(6, 12)  # 6组Conv-BN-ReLU 增长率12
        # [1,16,16,116]
        self.Transition2 = DenseNet.Transition(58)
        # [1,8,8,58]
        self.DenseBlock3 = DenseNet.DenseBlock(6, 12)  # 6组Conv-BN-ReLU 增长率12
        # [1,8,8,130]
        self.Transition3 = DenseNet.Transition(65)
        # [1,4,4,65]
        self.Classification = DenseNet.Classification(2)
        # [1,2]

    def call(self, input_tensor):
        x = self.Conv(input_tensor)
        x = self.Pool(x)
        x = self.DenseBlock1(x)
        x = self.Transition1(x)
        x = self.DenseBlock2(x)
        x = self.Transition2(x)
        x = self.DenseBlock3(x)
        x = self.Transition3(x)
        x = self.Classification(x)
        return x


def loss_func(y_true, y_pred):
    # 使用交叉熵损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return loss(y_true, y_pred)
