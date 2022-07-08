import time

import matplotlib.pyplot as plt
import tensorflow as tf
from utils import get_dataset

if __name__ == '__main__':
    try:
        Model = tf.keras.models.load_model("saved_model/my_model")
    except:
        print("Error: Could not find model")
        exit(0)

    test_data_list = get_dataset("../dataset/train", shuffle=True)
    start_time = time.time()
    for i in range(18):  # 取前18个进行测试
        img_raw = tf.io.read_file(test_data_list[i][0])  # 读图片数据流
        image = tf.image.decode_jpeg(img_raw, channels=1)  # 解码为灰度图
        image = tf.image.resize(image, [195, 195])  # 统一大小
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        image = tf.expand_dims(image, axis=0)
        model_output = Model(image)
        plt.subplot(3, 6, i + 1)  # 子图
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.image.decode_jpeg(img_raw, channels=3))
        # 若正确
        if tf.argmax(model_output, axis=1) == test_data_list[i][1]:
            if test_data_list[i][1] == 1:
                plt.xlabel("Pred: Perfect \nTrue: Perfect", color="green")
            elif test_data_list[i][1] == 0:
                plt.xlabel("Pred: Damaged \nTrue:Damaged", color="green")
        else:
            if test_data_list[i][1] == 0:
                plt.xlabel("Pred: Perfect \nTrue: Damaged", color="red")
            elif test_data_list[i][1] == 1:
                plt.xlabel("Pred: Damaged \nTrue: Perfect", color="red")
    plt.show()
    test_speed = 18/(time.time()-start_time)
    print("test speed: {:.3f}".format(test_speed))
