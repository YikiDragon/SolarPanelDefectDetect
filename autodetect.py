import os
import pathlib
from image_utils import correct, segment
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def segArrayPreprocess(image: np.ndarray, model='SVM'):
    if model == 'SVM':  # SVM
        HOG = cv2.HOGDescriptor((200, 200),  # winSize          # Hog算子
                                (80, 80),  # blockSize
                                (40, 40),  # blockStride
                                (40, 40),  # cellSize
                                9)  # nbins
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [200, 200])  # 统一大小
        image = image.numpy().astype(np.uint8)
        featrue_vector = tf.reshape(HOG.compute(image), [1, -1])
        return featrue_vector
    elif model == 'DenseNet':  # DenseNet
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [195, 195])  # 统一大小
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        image = tf.expand_dims(image, axis=0)
        return image


def judgment(result, model='SVM'):
    if model == 'SVM':
        if result < 0:  # SVM负数为缺陷
            return 0
        else:
            return 1  # SVM正数为完好
    elif model == 'DenseNet':
        return int(tf.argmax(result, axis=1))


if __name__ == '__main__':
    debug = False
    model = 'DenseNet'
    data_root = pathlib.Path('./photos')
    all_image_names = sorted(item.name for item in data_root.glob('*.JPG'))  # 获取文件名
    model_path = ''
    if model == 'SVM':
        model_path = "./SVM_Kernel/saved_model/my_model"
    elif model == 'DenseNet':
        model_path = "./DenseNet/saved_model/my_model"
    Model = tf.keras.models.load_model(model_path)  # 加载模型
    for image in all_image_names:
        print("image name: " + image)
        img_src = cv2.imread("photos/" + image)
        image_corrected = correct(img_src, debug=debug)
        draw_image = image_corrected.copy()
        seg_list = segment(image_corrected, seg_method=4, debug=debug)
        slices_num = len(seg_list)  # 切片总数
        for item in seg_list:  # 遍历任务
            point = item[0]
            img = item[1]
            img = segArrayPreprocess(img, model)  # 预处理
            result = Model(img)  # 送入模型
            conclusion = judgment(result, model)  # 统一判断数
            if conclusion == 0:  # 检测到缺陷
                # draw_image = cv2.rectangle(draw_image, tuple(point[0]), tuple(point[1]), (255, 0, 0), 10)     # 框选缺陷
                draw_image = cv2.drawMarker(draw_image, tuple(((point[0] + point[1]) / 2).astype(np.int32)),
                                            (255, 0, 0), markerType=1, markerSize=100, thickness=10)  # 缺陷画叉
        plt.xlabel("Detected")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(draw_image)
        plt.savefig("./result/" + image + "(detected).png", format="png", dpi=100,)
        plt.clf()
