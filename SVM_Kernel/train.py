import pickle
import sys
import cv2
import numpy as np
import os
import time
import datetime
from KernelSVM_model import SVMModel, kernel_gaussian, loss_func
from utils import get_dataset, load_and_preprocess_image, balanced_dataset

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 允许显存递增占用
import tensorflow as tf

if __name__ == '__main__':

    HOG = cv2.HOGDescriptor((200, 200),  # winSize
                            (80, 80),  # blockSize
                            (40, 40),  # blockStride
                            (40, 40),  # cellSize
                            9)  # nbins

    train = True
    test = False
    # update_num = None     # 无限循环
    update_num = 1000    # 更新次数
    # loss_th = -10.0
    C = 5              # 松弛因子
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_data_list = get_dataset('../dataset/train', shuffle=False)    # 数据列表获取
    train_data_list, test_data_list, train_PD_num, test_PD_num = balanced_dataset(train_data_list, label_ratio=4,
                                                                                  shuffle=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)  # 写入训练日志
    train_loss_recorder = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)  # 平均损失记录

    # start_time = time.time()  # 训练开始时间
    if train:
        """
        训练
        """
        data_x = np.array([], dtype=np.float32)
        data_y = np.array([], dtype=np.float32)
        for i in range(sum(train_PD_num)):
            data_x = np.append(data_x, HOG.compute(load_and_preprocess_image(train_data_list[i][0])))
            data_y = np.append(data_y, train_data_list[i][1])
        data_x = np.reshape(data_x, [sum(train_PD_num), -1])
        data_y = np.reshape(data_y, [sum(train_PD_num), -1])
        # 如果有上一次训练的优化器则加载优化器
        if os.path.exists("saved_model/optimizer.data"):
            with open("./saved_model/optimizer.data", "rb") as saved_optimizer:
                optimizer = pickle.load(saved_optimizer)
            # optimizer.set_weights(optimizer_weight)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": Find saved optimizer")
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": No saved optimizer")
        # 如果已有模型则加载模型
        if os.path.exists("./saved_model/my_model"):
            SVM = tf.keras.models.load_model("./saved_model/my_model")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": Find saved model")
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": No saved model, Continute......")
            SVM = SVMModel(data_x, data_y, kernel=kernel_gaussian)  # 新建模型
        SVM(tf.expand_dims(data_x[0], axis=0))
        if update_num is not None:
            for update_ord in range(update_num):
                # 梯度下降与权重更新
                with tf.GradientTape() as tape:
                    loss = loss_func(SVM, C)
                optimizer.apply_gradients(
                    zip(tape.gradient(loss, SVM.trainable_variables), SVM.trainable_variables))
                train_loss_recorder(loss)   # 记录损失
                if not test:
                    print('loss: {:.3f}, please wait...'.format(loss), end='\r')
                if test:
                    """
                    测试
                    """
                    pred = SVM(data_x)
                    true_num = tf.reduce_sum(tf.where(tf.sign(tf.transpose(pred)) == data_y, 1, 0))
                    accuracy = true_num / sum(train_PD_num) * 100
                    print('loss: {:.3f}, accuracy: {:.3f} please wait...'.format(loss, accuracy), end='\r')
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_recorder.result(), step=update_ord)
                print('ord:{}, loss: {:.3f}, please wait...'.format(update_ord, loss), end='\r')

            SVM.save("./saved_model/my_model")  # 保存模型
            print("Model saved")
            with open("./saved_model/optimizer.data", "wb") as saved_optimizer:
                pickle.dump(optimizer, saved_optimizer)  # 保存优化器
                print("optimizer_weights saved")
        else:
            while True:
                # 梯度下降与权重更新
                with tf.GradientTape() as tape:
                    loss = loss_func(SVM, C)
                optimizer.apply_gradients(
                    zip(tape.gradient(loss, SVM.trainable_variables), SVM.trainable_variables))
                if not test:
                    print('loss: {:.3f}, please wait...'.format(loss), end='\r')
                if test:
                    """
                    测试
                    """
                    pred = SVM(data_x)
                    true_num = tf.reduce_sum(tf.where(tf.sign(tf.transpose(pred)) == data_y, 1, 0))
                    accuracy = true_num / sum(train_PD_num) * 100
                    print('loss: {:.3f}, accuracy: {:.3f} please wait...'.format(loss, accuracy), end='\r')
                    if accuracy == 100:
                        # 保存模型与优化器参数
                        SVM.save("./saved_model/my_model")  # 保存模型
                        with open("./saved_model/optimizer.data", "wb") as saved_optimizer:
                            pickle.dump(optimizer, saved_optimizer)  # 保存优化器
                        print("Test accuracy Satisfied")
                        sys.exit()
                # if loss < loss_th:
                #     print('LastTime: loss: {:.3f}, finished'.format(loss), end='\r')
                #     break
            SVM.save("./saved_model/my_model")  # 保存模型
            print("Model saved")
            with open("./saved_model/optimizer.data", "wb") as saved_optimizer:
                pickle.dump(optimizer, saved_optimizer)  # 保存优化器
                print("optimizer_weights saved")

            # 保存模型与优化器参数

    # with open("optimizer.data", "wb") as saved_optimizer:
    #     pickle.dump(optimizer, saved_optimizer)  # 保存优化器
