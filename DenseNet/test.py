import time
import datetime
import tensorflow as tf
import model
import os
import sys
from utils import get_dataset, PR_Recorder, load_and_preprocess_image

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 允许显存递增占用
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'         # 是否使用NVIDIA显卡加速，-1禁用

if __name__ == '__main__':
    """
    模型初始化
    """
    Model = model.Model()  # 新建模型
    PR_recorder = PR_Recorder(2)
    # 如果已有模型则加载模型
    if os.path.exists("saved_model/my_model"):
        Model = tf.keras.models.load_model("saved_model/my_model")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": Find saved model")
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ": No saved model, Continute......")
        sys.exit()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    test_data_list = get_dataset('../dataset/train', shuffle=True)
    perfect_true_num = 0  # 完好正确数
    damaged_true_num = 0  # 缺陷正确数
    perfect_num = 0.001  # 动态统计完好标签数
    damaged_num = 0.001  # 动态统计缺陷标签数
    perfect_accuracy = 0.0  # 完好正确率
    damaged_accuracy = 0.0  # 缺陷正确率
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)  # 写入训练日志
    test_loss_recorder = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)  # 平均损失记录
    test_num = len(test_data_list)  # 测试集数
    total_time = 0
    for image_i in range(test_num):  # 遍历训练集
        image_test = tf.expand_dims(load_and_preprocess_image(test_data_list[image_i][0]), axis=0)
        image_test = tf.cast(image_test, dtype=tf.float32)
        start_time = time.time()
        model_output = Model(image_test)
        model_time = time.time() - start_time
        total_time += model_time
        PR_recorder.Record_output(test_data_list[image_i][1], tf.reshape(model_output, shape=[-1, 2]))  # PR数据记录
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
            print('Test Completed : %s [%d/%d] Accuracy:(Perfect: %.3f%% Damaged: %.3f%%) MeanFPS: %.3f' % (str(percent) + '%', (image_i + 1), test_num, perfect_accuracy, damaged_accuracy, test_num/total_time), end='\n')
        else:
            test_speed = 1 / model_time
            percent = round(1.0 * (image_i + 1) / test_num * 100, 2)
            print(
                'Testing.... Current progress : %s [%d/%d] Accuracy:(Perfect: %.3f%% Damaged: %.3f%%) RTFPS: '
                '%.3f' % (str(percent) + '%',
                                 (image_i + 1),
                                 test_num,
                                 perfect_accuracy,
                                 damaged_accuracy,
                                 test_speed), end='\r')
    PR_recorder.Generate_PR()
    PR_recorder.plot_PR_curve(ClassNameList=['Damaged', 'Perfect'], title="DenseNet P-R Curve", fig_path='plot')
    template = 'Loss: {}, Test accuracy: (Perfect: {:.3f}% Damaged: {:.3f}%)'
    # template = 'Epoch {}, Loss: {}, AP: {:.3f}%'
    print(template.format(test_loss_recorder.result(),
                          perfect_accuracy,
                          damaged_accuracy
                          ))
