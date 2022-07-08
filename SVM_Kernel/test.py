import pickle
import sys
import cv2
import numpy as np
import os
import time
import datetime
from KernelSVM_model import SVMModel, kernel_gaussian, loss_func
from utils import get_dataset, load_and_preprocess_image, PR_Recorder

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 允许显存递增占用
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'         # 是否使用NVIDIA显卡加速，-1禁用
import tensorflow as tf

if __name__ == '__main__':

    HOG = cv2.HOGDescriptor((200, 200),  # winSize
                            (80, 80),  # blockSize
                            (40, 40),  # blockStride
                            (40, 40),  # cellSize
                            9)  # nbins
    test_num = 8592  # 8592
    test_data_list = get_dataset('../dataset/train', shuffle=False)  # 数据列表获取
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    PR_recorder = PR_Recorder(2)
    """
    训练
    """
    data_x = np.array([], dtype=np.float32)
    data_y = np.array([], dtype=np.float32)
    for i in range(test_num):
        data_x = np.append(data_x, HOG.compute(load_and_preprocess_image(test_data_list[i][0])))
        data_y = np.append(data_y, test_data_list[i][1])
    data_x = np.reshape(data_x, [test_num, -1])
    data_y = np.reshape(data_y, [test_num, -1])

    # 如果已有模型则加载模型
    if os.path.exists("./saved_model/my_model"):
        SVM = tf.keras.models.load_model("./saved_model/my_model")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": Find saved model")
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": No saved model, Continute......")
        sys.exit()

    pred = SVM(data_x)  # 全体预测
    true_num = tf.reduce_sum(tf.where(tf.sign(pred) == data_y, 1, 0))
    accuracy = true_num / test_num * 100
    print('accuracy: {:.3f} test_num: {}'.format(accuracy, test_num), end='\n')
    # PR曲线
    print("Generating PR Curve Please wait...")
    data_y = np.where(data_y == -1, 0, data_y)
    total_time = 0
    for i in range(test_num):
        start_time = time.time()
        model_output = SVM(tf.expand_dims(data_x[i], axis=0))
        model_time = time.time()-start_time
        print("Test RTFPS: {:.3f}, {:.3f}% [{}/{}]".format(1/model_time, (i+1)/test_num*100, i+1, test_num), end="\r")
        total_time += model_time
        if model_output < 0:
            pred_label = tf.cast([1.0, 0.0], tf.float32)
        else:
            pred_label = tf.cast([0.0, 1.0], tf.float32)
        PR_recorder.Record_output(data_y[i], tf.reshape(pred_label, shape=[-1, 2]))  # PR数据记录
    print("Test MeanFPS: {:.3f}, 100%".format(test_num/total_time))
    PR_recorder.Generate_PR()
    PR_recorder.plot_PR_curve(ClassNameList=['Damaged', 'Perfect'], title="SVM P-R Curve", fig_path='plot')
