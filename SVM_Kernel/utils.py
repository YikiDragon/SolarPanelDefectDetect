import pathlib
import random
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def gamma_norm(image, gamma=0.5):
    '''
    伽马归一化
    :param image:   图片
    :param gamma:   伽马值
    :return:        处理后图片
    '''
    image_temp = np.array(image, copy=True)
    image_temp = image_temp.astype(np.float32)  # 防溢出
    image_temp = image_temp / 255  # 归一化
    image_temp = 255 * np.power(image_temp, gamma)  # 指数并复原
    image_temp = np.where(image_temp < 256, image_temp, 255)
    image_temp = image_temp.astype(np.uint8)
    return image_temp


def get_dataset(root_dir, shuffle=False):
    """
    读数据集中各图片路径与标签
    输入：root_dir 根目录文件夹
    输出：all_image_paths 各图片的路径
         all_image_labels 各图片的标签
    """
    data_root = pathlib.Path(root_dir)  # 设置数据根目录
    # for item in data_root.iterdir():
    #     print(item)  # 遍历根目录下的项目
    all_image_paths = list(data_root.glob('*/*'))  # 生成图片路径列表
    all_image_paths = [str(path) for path in all_image_paths]  # 转为字符串列表
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())  # 获取标签
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    label_to_index["damaged"] = -1  # 损坏的标签为-1，完好的标签为1
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]  # 根据图片所在文件夹名划分标签
                        for path in all_image_paths]
    data_list = []
    for i in range(len(all_image_paths)):
        data_list.append([all_image_paths[i], all_image_labels[i]])
    if shuffle:
        random.shuffle(data_list)
    return data_list


def balanced_dataset(data_list, sample_refer=0, sample_rate=0.8, label_ratio=1, shuffle=False):
    """
    :param data_list:       数据标签列表
    :param sample_refer:    取样参考,0参考损坏集，1参考完好集
    :param sample_rate:     取样率
    :param label_ratio:     标签比
    :param shuffle:         是否打乱
    :return:    train_dataset 平衡训练集
                test_dataset  平衡测试集
    """
    perfect_list = []
    damaged_list = []
    train_perfect_num = 0
    train_damaged_num = 0
    for data in data_list:
        if data[1] == 1:  # 完好
            perfect_list.append(data)
        elif data[1] == -1:  # 损坏
            damaged_list.append(data)
    if sample_refer == 0:  # 以损坏集作为参考
        train_damaged_num = int(len(damaged_list) * sample_rate)
        train_perfect_num = int(train_damaged_num * label_ratio)
    elif sample_refer == 1:  # 以完好集作为参考
        train_perfect_num = int(len(perfect_list) * sample_rate)
        train_damaged_num = int(train_perfect_num * label_ratio)
    random.shuffle(perfect_list)
    random.shuffle(damaged_list)
    # 划分训练集
    train_perfect_list = perfect_list[:train_perfect_num]
    train_damaged_list = damaged_list[:train_damaged_num]
    # 划分测试集
    test_perfect_list = perfect_list[train_perfect_num:]
    test_damaged_list = damaged_list[train_damaged_num:]
    train_dataset = train_perfect_list + train_damaged_list  # 合成训练集
    test_dataset = test_perfect_list + test_damaged_list  # 合成测试集
    if shuffle:
        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
    print("\t平衡训练集: %d \t平衡测试集: %d" % (len(train_dataset), len(test_dataset)))
    print("完好数: \t%d \t\t%d" % (len(train_perfect_list), len(test_perfect_list)))
    print("损坏数: \t%d \t\t%d" % (len(train_damaged_list), len(test_damaged_list)))
    return train_dataset, \
           test_dataset, \
           [len(train_perfect_list), len(train_damaged_list)], \
           [len(test_perfect_list), len(test_damaged_list)]


def load_and_preprocess_image(path, image_size=None):
    """
    根据路径读取图片张量并统一大小
    """
    if image_size is None:
        image_size = [200, 200]
    img_raw = tf.io.read_file(path)  # 读图片数据流
    image = tf.image.decode_jpeg(img_raw, channels=1)  # 解码为灰度图
    image = tf.image.resize(image, image_size)  # 统一大小
    image = image.numpy().astype(np.uint8)
    # image = tf.reshape(image, [1, 400, 400])
    # image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))  # 图片数据归一化
    # return gamma_norm(image)    # gamma校正
    return image

class PR_Recorder:

    def __init__(self, class_num=7):
        self.Class_num = class_num  # 1种背景+6种待测目标
        #
        self.TP_Flag = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]
        # 预测各类分数
        self.scores = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]
        # TP+FN
        self.TP_FN = [0.0 for c in range(self.Class_num)]
        # 精准率:所有数据中正例预测比例
        self.Precision = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]
        # 召回率:预测为正中正确预测比例
        self.Recall = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]
        # PR曲线积分
        self.AP = [0.0 for c in range(self.Class_num)]

    def Record_output(self, true_labels, pred_scores, gtbboxes=None, pred_bboxes=None, IoU_th=None):
        '''
        对单张图片记录输出
        :param true_labels:     真实标签    [n,1,1]
        :param pred_scores:     预测分数    [n,m,c]   n个位置可能产生m个不同形状的预测框，每个框有c个预测类别
        :param gtbboxes:        真实框     [n,1,4]
        :param pred_bboxes:     预测框     [n,m,4]    n个位置可能产生m个不同形状的预测框
        :param IoU_th:          IoU判定阈值
        :return:
        '''
        def IoU(bbox1, bbox2):  # bbox格式[ymin,xmin,ymax,ymin]
            """
            计算IoU
            :param bbox1: [4,] Tensor
            :param bbox2: [4,] Tensor
            :return IoU:
            """
            A_j_B_w = tf.maximum(tf.minimum(bbox1[3], bbox2[3]) - tf.maximum(bbox1[1], bbox2[1]), 0)
            A_j_B_h = tf.maximum(tf.minimum(bbox1[2], bbox2[2]) - tf.maximum(bbox1[0], bbox2[0]), 0)
            A_j_B = A_j_B_w * A_j_B_h
            A = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
            B = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
            A_b_B = A + B - A_j_B
            IoU = A_j_B / A_b_B
            return IoU

        # 对目标检测问题，根据IoU筛选TP
        # 一张图片可能有多个gtbbox
        if IoU_th != None:
            for i in range(self.Class_num):  # 每个类别都进行扫描, i为当前扫描的类别编号
                index_gt = tf.where(true_labels == i)  # 获取gtlabels里的为该类的索引
                index_pr = tf.where(np.argmax(pred_scores, axis=-1) == i)   # 获取prlabels里的为该类的索引
                if index_gt.shape[0] != 0:                                  # 存在当前类的真实标签
                    self.TP_FN[i] += index_gt.shape[0]                      # 当前类的真实总数
                    if index_pr.shape[0] != 0:      # 存在当前类的预测标签
                        i_prscores = tf.reshape(tf.gather(pred_scores, index_pr), [-1])     # 筛选出为当前类的分数
                        i_gtbboxes = tf.reshape(tf.gather(gtbboxes, index_gt), [-1, 4])     # 当前类的gtbboxes
                        i_prbboxes = tf.reshape(tf.gather(pred_bboxes, index_pr), [-1, 4])  # 当前类的prbboxes
                        TP_mask = [0.0 for i_prbbox in i_prbboxes]  # 生成TP_mask向量
                        for i_gtbbox in i_gtbboxes:
                            for j, i_prbbox in enumerate(i_prbboxes):   # 当前类的gt和pr的IoU大于阈值视为TP
                                if IoU(i_prbbox, i_gtbbox) >= IoU_th:  # IoU大于阈值视为TP
                                    TP_mask[j] = 1.0  # 标为TP
                                    break             # 有一个视为TP的预测框就认为该gtbbox被检测到了,检测下一个gtbbox
                        # 最后形成一个该类prbbox数量的TP_Flag
                        self.TP_Flag[i] = tf.concat([self.TP_Flag[i], TP_mask], axis=0)
                        # 该类prbbox数量的该类分数
                        self.scores[i] = tf.concat([self.scores[i], i_prscores[i]], axis=0)
        # 对图像分类问题，根据分数筛选TP
        # 一张图片只有一个label
        else:
            for i in range(self.Class_num):  # 每个类别都进行扫描, i为当前扫描的类别编号
                if true_labels == i:      # 存在当前类的真实标签
                    self.TP_FN[i] += 1  # 当前类的真实总数
                    if np.argmax(pred_scores, axis=-1) == i:  # 存在当前类的预测标签
                        TP_mask = [1.0]   # 预测为TP
                        self.TP_Flag[i] = tf.concat([self.TP_Flag[i], TP_mask], axis=0)
                        self.scores[i] = tf.concat([self.scores[i], pred_scores[:, i]], axis=0)
                    else:
                        TP_mask = [0.0]   # 预测错误
                        self.TP_Flag[i] = tf.concat([self.TP_Flag[i], TP_mask], axis=0)
                        self.scores[i] = tf.concat([self.scores[i], pred_scores[:, i]], axis=0)


    def Generate_PR(self):
        for i in range(self.Class_num):  # 每个类别都进行扫描, i为当前扫描的类别编号
            # if i == 0:
            #     continue
            i_TP_mask = self.TP_Flag[i]     # 第i类的Flag列表
            i_scores = self.scores[i]       # 第i类的得分列表
            idx_sort = tf.argsort(i_scores, direction='DESCENDING')  # 降序排列
            i_TP_mask = tf.reshape(tf.gather(i_TP_mask, idx_sort), [-1])
            ACC_TP = 0.0
            TP_FP = 0.0
            TP_FN = self.TP_FN[i]  # 第i类TP+FN总数,非0
            for mask in i_TP_mask:  # 累加操作
                ACC_TP += mask  # 如果是TP则加1,FP则加0
                TP_FP += 1  # 扫描一个pred_box就加1
                precision = ACC_TP / TP_FP
                recall = ACC_TP / TP_FN
                self.Precision[i] = tf.concat([self.Precision[i], [precision]], axis=0)  # 添加精确率
                self.Recall[i] = tf.concat([self.Recall[i], [recall]], axis=0)  # 添加召回率
        return self.Precision, self.Recall

    # PR曲线积分计算AP
    def compute_AP(self):
        for i in range(self.Class_num):
            # if i == 0:
            #     continue
            AP = 0.0
            Precision = self.Precision[i]
            Recall = self.Recall[i]
            if Precision.shape[0] == 0:
                continue
            point_a_y = Precision[0]
            point_a_x = Recall[0]
            for j, point_b_x in enumerate(Recall):
                point_b_y = Precision[j]
                width = point_b_x - point_a_x
                height = max(point_b_y, point_a_y)
                AP += width * height
                point_a_x = point_b_x
                point_a_y = point_b_y
            self.AP[i] = float(AP)
        return self.AP

    def plot_PR_curve(self, ClassNameList, title, fig_path=None):
        CLASSES = ClassNameList
        AP =  self.compute_AP()
        colors = dict()  # 颜色字典
        Precison = self.Precision
        Recall = self.Recall
        CLS_AP = []
        template = '{}(AP={:.3f})'
        for i, CLASS in enumerate(CLASSES):
            CLS_AP.append(template.format(CLASS, AP[i]))
            colors[i] = (random.random(), random.random(), random.random())
            # if i == 0:
            #     continue
            plt.plot(Recall[i], Precison[i], color=colors[i])
            plt.title(title)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            # plt.text(0, 0.8, "AP=")
            # plt.text(0, 0.9, str(self.compute_AP()))
        plt.xlim(0.0, 1.1)
        plt.ylim(0.0, 1.1)
        plt.legend(CLS_AP)
        plt.grid()
        if fig_path is not None:
            plt.savefig(fig_path + "/PR_Curve_" + time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + ".svg", format="svg")
        plt.show()

    def reset_state(self):
        self.TP_Flag = [tf.convert_to_tensor([], dtype=tf.int32) for c in range(self.Class_num)]
        self.scores = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]
        self.TP_FN = [0 for c in range(self.Class_num)]
        self.Precision = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]
        self.Recall = [tf.convert_to_tensor([], dtype=tf.float32) for c in range(self.Class_num)]