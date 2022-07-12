import pickle
import random
import sys
import numpy as np
import model
import os
import time
import datetime
import tensorflow as tf
from utils import get_dataset, balanced_dataset, load_and_preprocess_image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 允许显存递增占用

if __name__ == '__main__':

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    train = True       # 是否训练
    test = True         # 是否测试
    EPOCHS = 75  # 世代数，遍历全部训练集的次数
    batch_size = 32  # 批大小，batch训练集大小
    Itertion = 60  # 迭代数，训练集内更新权重的次数（batch数）
    image_size = [195, 195]
    learning_rate = 0.001  # 选择学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate)   # Adam优化器
    # optimizer = tf.keras.optimizers.SGD(learning_rate)      # 经典梯度下降优化器
    # shuffle_rate = 5000  # 混乱度，打乱数据集

    train_num = batch_size * Itertion  # 单次epoch的训练数 #6672

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)  # 写入训练日志
    train_loss_recorder = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)  # 平均损失记录
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)  # 写入训练日志
    test_loss_recorder = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)  # 平均损失记录
    """
    模型初始化
    """
    Model = model.Model()  # 新建模型
    # 如果已有模型则加载模型
    if os.path.exists("saved_model/my_model"):
        Model = tf.keras.models.load_model("saved_model/my_model")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": Find saved model")
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": No saved model, Continute......")

    """
    获取数据集
    """
    train_data_list = get_dataset('../dataset/train', shuffle=False)
    # 平衡数据集
    train_data_list, test_data_list, train_PD_num, test_PD_num = balanced_dataset(train_data_list, label_ratio=4, shuffle=True)
    # start_time = time.time()  # 训练开始时间

    '''
    训练
    '''
    if train:
        """
        优化器初始化
        """
        # 如果有上一次训练的优化器则加载优化器
        if os.path.exists("optimizer.data"):
            with open("optimizer.data", "rb") as saved_optimizer:
                optimizer = pickle.load(saved_optimizer)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": Find saved optimizer")
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": No saved optimizer")

        """
        世代
        """
        for epoch in range(EPOCHS):

            random.shuffle(train_data_list)  # 每一个世代打乱训练集
            train_loss_recorder.reset_states()  # 训练损失记录器重置
            test_loss_recorder.reset_states()   # 测试损失记录器重置

            total = train_num  # 训练总数
            batch_images = np.array([], dtype=np.float32)  # 图片batch
            batch_labels = np.array([], dtype=np.float32)  # 标签batch
            batch_start_time = time.time()  # batch开始时间
            train_speed = 0.0
            loss = 0.0
            for image_i in range(train_num):  # 遍历训练集
                # 合成batch数据集
                batch_images = np.append(batch_images, load_and_preprocess_image(train_data_list[image_i][0], image_size))
                batch_labels = np.append(batch_labels, train_data_list[image_i][1])
                # 单个batch数据合成完成
                if (image_i + 1) % batch_size == 0:
                    batch_images = tf.reshape(batch_images, [batch_size]+image_size+[1])  # 变回矩阵
                    batch_labels = tf.reshape(batch_labels, [batch_size, -1])  # 变回矩阵
                    # 高斯核变换
                    # batch_images = model.gaussian_kernel(batch_images)
                    # 启动梯度带记录计算过程
                    with tf.GradientTape() as tape:
                        # loss = 0.0  # 合损失归零
                        # for batch_idx in range(batch_size):  # 第几个图片数据
                        # with tape.stop_recording():  # 中间过程不予记录
                        # single_image = tf.expand_dims(batch_images[batch_idx], axis=0)  # 样本图片
                        # single_label = batch_labels[batch_idx]  # 样本标签
                        # model_output = Model(batch_images)  # 样本切片图片输入模型进行计算
                        # loss += Model.loss_func(model_output, batch_labels)
                        # loss = loss / batch_size  # 单batch损失
                        model_output = Model(batch_images)
                        loss = model.loss_func(batch_labels, model_output)
                    # 梯度求解与权重更新
                    optimizer.apply_gradients(
                        zip(tape.gradient(loss, Model.trainable_variables), Model.trainable_variables))
                    train_loss_recorder(loss)

                    with train_summary_writer.as_default():
                        tf.summary.scalar('train_loss', train_loss_recorder.result(), step=epoch)

                    # 清空batch
                    batch_images = []
                    batch_labels = []
                    batch_bboxes = []
                    batch_end_time = time.time()  # batch结束时间
                    train_speed = batch_size / (batch_end_time - batch_start_time)
                    batch_start_time = batch_end_time

                if (image_i + 1) == total:
                    percent = 100.0
                    print('Epoch %d - Current progress : %s [%d/%d] Loss: %.3f' % (epoch, str(percent) + '%',
                                                                        (image_i + 1),
                                                                        total,
                                                                        train_loss_recorder.result()), end='\n')
                else:
                    percent = round(1.0 * (image_i + 1) / total * 100, 2)
                    print('Epoch %d (Training)- Current progress : %s [%d/%d] Train Speed : %.3f data/s Loss : %.4f' % (
                        epoch, str(percent) + '%', (image_i + 1), total, train_speed, loss), end='\r')
        # 保存模型与优化器参数
        Model.save("./saved_model/my_model")  # 保存模型
        print("model saved")
        # with open("optimizer.data", "wb") as saved_optimizer:
        #     pickle.dump(optimizer, saved_optimizer)  # 保存优化器
        #     print("saved: optimizer")
    """
    测试
    """
    if test:
        # test_data_list = get_dataset("./dataset/test")
        perfect_true_num = 0  # 完好正确数
        damaged_true_num = 0  # 缺陷正确数
        perfect_num = 0.001  # 动态统计完好标签数
        damaged_num = 0.001  # 动态统计缺陷标签数
        perfect_accuracy = 0.0  # 完好正确率
        damaged_accuracy = 0.0  # 缺陷正确率
        test_perfect_num, test_damaged_num = test_PD_num  # 测试完好集数，缺陷集数
        test_num = len(test_data_list)  # 测试集数
        for image_i in range(test_num):  # 遍历训练集
            start_time = time.time()
            image_test = tf.expand_dims(load_and_preprocess_image(test_data_list[image_i][0]), axis=0)
            image_test = tf.cast(image_test, dtype=tf.float32)
            model_output = Model(image_test)
            loss = model.loss_func(test_data_list[image_i][1], model_output)
            test_loss_recorder(loss)
            with test_summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss_recorder.result(), step=image_i)
            if tf.argmax(model_output, axis=1) == test_data_list[image_i][1]:  # 预测正确
                if test_data_list[image_i][1] == 1:  # 检测到完好
                    perfect_true_num += 1
                    perfect_num += 1
                elif test_data_list[image_i][1] == 0:  # 检测到缺陷
                    damaged_true_num += 1
                    damaged_num += 1
            else:  # 预测错误
                if test_data_list[image_i][1] == 1:  # 完好的标签
                    perfect_num += 1
                elif test_data_list[image_i][1] == 0:  # 缺陷的标签
                    damaged_num += 1
            perfect_accuracy = perfect_true_num / perfect_num * 100
            damaged_accuracy = damaged_true_num / damaged_num * 100
            if (image_i + 1) == test_num:
                percent = 100.0
                print('Test Completed : %s [%d/%d]' % (str(percent) + '%', (image_i + 1), test_num), end='\n')
            else:
                test_speed = 1 / (time.time() - start_time)
                percent = round(1.0 * (image_i + 1) / test_num * 100, 2)
                print(
                    'Testing.... Current progress : %s [%d/%d] Accuracy:(Perfect: %.3f%% Damaged: %.3f%%) Test speed: '
                    '%.3f data/s' % (str(percent) + '%',
                                     (image_i + 1),
                                     test_num,
                                     perfect_accuracy,
                                     damaged_accuracy,
                                     test_speed), end='\r')

        template = 'Loss: {}, Test accuracy: (Perfect: {:.3f}% Damaged: {:.3f}%)'
        # template = 'Epoch {}, Loss: {}, AP: {:.3f}%'
        print(template.format(test_loss_recorder.result(),
                              perfect_accuracy,
                              damaged_accuracy
                              ))
        if damaged_accuracy > 98 and perfect_accuracy > 90:
            # 保存模型与优化器参数
            Model.save("./saved_model/my_model")  # 保存模型
            with open("optimizer.data", "wb") as saved_optimizer:
                pickle.dump(optimizer, saved_optimizer)  # 保存优化器
                print("saved: optimizer")
            print("Test accuracy > 98%")
            sys.exit()

