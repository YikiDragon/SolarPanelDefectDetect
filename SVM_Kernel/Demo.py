import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import load_and_preprocess_image, get_dataset
import train

if __name__ == '__main__':
    try:
        Model = tf.keras.models.load_model("./saved_model/my_model")
        print("Find saved model")
    except:
        print("Error: Could not find model")
        exit(0)

    test_data_list = get_dataset("../dataset/train", shuffle=True)  # 获取数据及标签，打乱
    HOG = cv2.HOGDescriptor((200, 200),  # winSize          # Hog算子
                            (80, 80),  # blockSize
                            (40, 40),  # blockStride
                            (40, 40),  # cellSize
                            9)  # nbins
    start_time = time.time()
    for i in range(18):  # 取前18个进行测试
        image = load_and_preprocess_image(test_data_list[i][0], image_size=[200, 200])  # 输入图片路径解析成合适大小的图片
        hog_vector = tf.reshape(HOG.compute(image), [1, -1])    # 计算HoG特征向量并reshape到大小[1,576]
        model_output = Model(hog_vector)                        # 预测，正数是完好晶片，负数是缺陷晶片
        plt.subplot(3, 6, i + 1)  # 子图
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.image.decode_png(tf.io.read_file(test_data_list[i][0])), cmap=plt.get_cmap('gray'))
        # 若正确
        if (model_output < 0 and test_data_list[i][1] == -1) or (
                model_output > 0 and test_data_list[i][1] == 1):
            if test_data_list[i][1] == 1:
                plt.xlabel("Pred: Perfect \nTrue: Perfect", color="green")
            elif test_data_list[i][1] == -1:
                plt.xlabel("Pred: Damaged \nTrue:Damaged", color="green")
        else:
            if test_data_list[i][1] == 1:
                plt.xlabel("Pred: Perfect \nTrue: Damaged", color="red")
            elif test_data_list[i][1] == -1:
                plt.xlabel("Pred: Damaged \nTrue: Perfect", color="red")
    plt.show()
    test_speed = 18/(time.time()-start_time)
    print("test speed: {:.3f}".format(test_speed))