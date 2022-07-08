import tensorflow as tf
from tensorflow.keras.layers import Multiply, Flatten, Subtract, Dense


# 自定义SVM层
class SVMLayer(tf.keras.layers.Layer):
    def __init__(self, saved_X, saved_Y, kernel):
        super(SVMLayer, self).__init__()
        self.kernel_method = kernel  # 所选核方法
        # 新建变动参数
        self.a = self.add_weight("alpha", shape=[saved_X.shape[0], 1])  # 有几行x数据就有几个a参数
        self.b = self.add_weight("bias", shape=[1, 1])  # 只有一个b参数
        # 新建固定参数
        self.X = self.add_weight("saved_X", shape=saved_X.shape, trainable=False)  # X
        self.Y = self.add_weight("saved_Y", shape=[saved_X.shape[0], 1], trainable=False)  # 有几行x数据就有几个y标签
        self.K = self.add_weight("saved_K", shape=[saved_X.shape[0], saved_X.shape[0]], trainable=False)  # K矩阵
        # 将X,Y,K数据置入权重列表更新，用于model.save保存
        weight_list = self.get_weights()
        weight_list[2] = saved_X  # 待存储的X
        weight_list[3] = saved_Y  # 待存储的Y
        weight_list[4] = kernel(saved_X, saved_X)  # 根据X直接生成核矩阵，方便进行训练
        self.set_weights(weight_list)  # 更新权重，将X，Y，K放入存储

    def build(self, inputshape):  # 初始化层
        print('SVM inputshape:' + str(inputshape))

    #     # 新建待训练参数
    #     self.a = self.add_weight("alpha", shape=[self.X.shape[0], 1])              # 有几行x数据就有几个a参数
    #     self.b = self.add_weight("bias", shape=[1, 1])                              # 只有一个b参数

    def call(self, inputs):  # 调用层
        return tf.transpose(tf.add(tf.reduce_sum(self.a * self.Y * self.kernel_method(self.X, inputs), axis=0), self.b))


# 自定义SVM模型
class SVMModel(tf.keras.Model):
    def __init__(self, saved_X, saved_Y, kernel):
        super(SVMModel, self).__init__()
        self.SVMLayer = SVMLayer(saved_X, saved_Y, kernel)

    def call(self, inputs):
        x = self.SVMLayer(inputs)
        return x


def kernel_gaussian(x_data, prediction_grid, gamma=10.0):
    '''

    :param x_data:              参与生成的X数据矩阵
    :param prediction_grid:     待预测的单个x数据向量
    :param gamma:               方差倒数
    :return:
    '''
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                          tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(-gamma, tf.abs(pred_sq_dist)))
    return pred_kernel


def loss_func(Model: SVMModel, C):
    a, b, X, Y, K = Model.weights
    first_term = tf.reduce_sum(a)  # sigma(a)
    aiaj = tf.matmul(a, tf.transpose(a))  # a参数交叉相乘矩阵
    yiyj = tf.matmul(Y, tf.transpose(Y))  # y标签交叉相乘矩阵
    aiajyiyj = tf.multiply(aiaj, yiyj)
    second_term = tf.multiply(tf.reduce_sum(tf.multiply(aiajyiyj, K)), 0.5)
    loss = tf.subtract(second_term, first_term) + C * tf.reduce_sum(tf.maximum(0, 1.0 - Y * Model(X)))
    return loss


'''
无监督学习部分
'''


def loss_func_tsvm(Model: SVMModel, Cl, Cu, y_u_pred, Y_u, split_Cu=False):
    '''
    :param Model:   SVM模型
    :param Cl:      全监督权重
    :param Cu:      无监督权重
    :param y_u_pred:  无监督模型预测值
    :param split_Cu:  是否拆分无监督权重
    :return: loss_tsvm 半监督svm损失
    '''
    loss_svm = loss_func(Model, Cl)  # 全监督部分损失
    # Y = tf.where(y_u_pred > 0, 1, -1)  # 根据无监督部分预测结果进行伪标记
    if not split_Cu:
        # 不使用拆分Cu
        # y_pred = SVMModel(Du)                   # 对无监督部分进行预测
        hinge_u = tf.reduce_sum(tf.maximum(0, 1.0 - Y_u * y_u_pred))
        loss_tsvm = loss_svm + Cu * hinge_u
        return loss_tsvm
    elif split_Cu:
        # 拆分Cu,以平衡无监督样本影响
        index_u_p = tf.where(Y_u == 1)      # 获取正伪标记索引
        index_u_n = tf.where(Y_u == -1)     # 获取负伪标记索引
        u_p = index_u_p.shape[0]  # 正伪标记数量
        u_n = index_u_n.shape[0]  # 负伪标记数量
        Cu_n = Cu * u_p / (u_n + u_p)  # 以Cu作为Cu_n
        Cu_p = Cu * u_n / (u_n + u_p)  # Cu_p
        hinge_u_p = tf.reduce_sum(
            tf.maximum(0, 1.0 - tf.gather_nd(Y_u, index_u_p) * tf.gather_nd(y_u_pred, index_u_p)))
        hinge_u_n = tf.reduce_sum(
            tf.maximum(0, 1.0 - tf.gather_nd(Y_u, index_u_n) * tf.gather_nd(y_u_pred, index_u_n)))
        loss_tsvm = loss_svm + Cu_p * hinge_u_p + Cu_n * hinge_u_n
        return loss_tsvm


def exchange_Y_u(y_u_pred, Y_u):
    '''
    判断是否有可能错误的两个伪标记，并将他们交换
    :param y_u_pred:    # 预测值
    :param Y_u:         # 伪标记
    :return: y_u_pred   # 原预测值
             Y_u        # 交换后的伪标记
             Flag       # 是否进行了交换
    '''
    ksi = tf.maximum(0, 1.0 - Y_u * y_u_pred)  # 松弛向量
    YiYj = tf.matmul(Y_u, tf.transpose(Y_u))  # Yi*Yj交叉矩阵
    ksiksi = tf.concat([
        tf.expand_dims(tf.ones(ksi.shape[0]) * ksi, axis=-1),
        tf.expand_dims(tf.transpose(tf.ones(ksi.shape[0]) * ksi), axis=-1)], axis=-1)
    Y_u = Y_u.numpy()
    # 提取满足要求的序列索引
    YiYj_index = tf.where(
        (YiYj < 0) & (ksiksi[:, :, 0] > 0) & (ksiksi[:, :, 1] > 0) & (tf.reduce_sum(ksiksi, axis=-1) > 2))
    if YiYj_index.shape[0] == 0:
        return y_u_pred, tf.cast(Y_u, tf.float32), False
    i, j = YiYj_index[0]
    Y_u[i] = -Y_u[i]
    Y_u[j] = -Y_u[j]
    print('exchange happened')
    return y_u_pred, tf.cast(Y_u, tf.float32), True

    # for i in range(Y_u.shape[0]):

# # Create an instance of the model
# x = tf.eye(3,3)
# y = tf.cast([[1],[-1],[1]],tf.float32)
# SVM = SVMModel(x, y, kernel=kernel_gaussian)
