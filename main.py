import cmd
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tf不打印调试信息
import json
import time
import sys
from alive_progress import alive_bar
from image_utils import correct, segment
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SVM_path = "./SVM_Kernel/saved_model/my_model"
DenseNet_path = "./DenseNet/saved_model/my_model"


class DetectConsole(cmd.Cmd):
    def __init__(self):
        super(DetectConsole, self).__init__()
        self.prompt = 'detecter >'
        self.conf = {"folder": "./", "model": 'SVM', "image": ""}
        self.all_folder_names = []
        self.all_image_names = []
        self.Model = tf.keras.models.Model

    def do_show(self, arg):
        'Usage:\n  show <command>\n\nDescription:\n  Show information or options available for the current operation.\n'
        if arg == 'model':
            print("Available models:\n0.SVM\n1.DenseNet")
        elif arg == 'folder':
            if len(self.all_folder_names) == 0:
                print("\033[1;33mNo folders available in the root folder!\033[0m\n")
            else:
                print("Available folders:\n")
                for i, item in enumerate(self.all_folder_names):
                    print("{} - {}".format(i, item))
        elif arg == 'image':
            if len(self.all_image_names) == 0:
                print("\033[1;33mNo pictures available in the current folder!\033[0m\n")
            else:
                print("Available images:\n")
                for i, item in enumerate(self.all_image_names):
                    print("{} - {}".format(i, item))

    def do_set(self, arg):
        'Set the necessary parameters.\nExample: set [options] <param>'
        arg = arg.split()
        # 指令正常
        if len(arg) == 2:
            if arg[0] == "folder":  # 选择图片文件夹
                try:
                    # 根据文件夹名寻找
                    if (os.path.exists('./' + arg[1]) and os.path.isdir('./' + arg[1])):  # 如果文件夹存在
                        self.conf["folder"] = arg[1] + '/'  # 装载
                        self.conf["image"] = ""  # 图片可能不在这个文件夹里了，消掉
                        self.updatePrompt()  # 更新指示符
                        self.updateItemList()  # 更新项目列表
                        self.check_images()  # 如果文件夹没有图片，打印警告信息
                        self.update_config()
                    # 根据文件夹序号寻找
                    elif (os.path.exists('./' + self.all_folder_names[int(arg[1])]) and os.path.isdir(
                            './' + self.all_folder_names[int(arg[1])])):
                        self.conf["folder"] = self.all_folder_names[int(arg[1])] + '/'  # 装载
                        self.conf["image"] = ""  # 图片可能不在这个文件夹里了，消掉
                        self.updatePrompt()  # 更新指示符
                        self.updateItemList()  # 更新项目列表
                        self.check_images()  # 如果文件夹没有图片，打印警告信息
                        self.update_config()
                    else:
                        print("\033[1;31m" + time.strftime(
                            "%Y-%m-%d %H:%M:%S ") + "ERROR: not a folder or the folder does not exists" + "\033[0m")
                        return 0
                except:
                    print("\033[1;31m" + time.strftime(
                        "%Y-%m-%d %H:%M:%S ") + "ERROR: An unknown error occurred while selecting a folder" + "\033[0m")

            elif arg[0] == "model":  # 选择模型
                if arg[1] == '0' or arg[1] == 'SVM':  # SVM模型
                    self.conf['model'] = 'SVM'  # 模型为0
                    ##### SVM ######
                    print(time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: Loading detection model: SVM Please Wait...")
                    self.Model = tf.keras.models.load_model(SVM_path)  # 加载模型
                    print(time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: Change detection model to: SVM")
                    self.updatePrompt()  # 更新指示符
                    self.update_config()
                elif arg[1] == '1' or arg[1] == 'DenseNet':  # DenseNet模型
                    self.conf['model'] = 'DenseNet'  # 模型为1
                    ###### DenseNet ######
                    print(
                        time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: Loading detection model: DenseNet Please Wait...")
                    self.Model = tf.keras.models.load_model(DenseNet_path)  # 加载模型
                    print(time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: Change detection model to: DenseNet")
                    self.updatePrompt()  # 更新指示符
                    self.update_config()
                else:  # 没有这种模型
                    print("\033[1;31m" + time.strftime(
                        "%Y-%m-%d %H:%M:%S ") + "ERROR: no matching model" + "\033[0m")

            elif arg[0] == "image":  # 选择图片名
                try:
                    # 按文件名
                    if os.path.exists('./' + self.conf["folder"] + arg[1]):  # 如果图片文件存在
                        self.conf['image'] = arg[1]  # 装载
                        self.updatePrompt()  # 更新指示符
                        self.updateItemList()  # 更新项目列表
                        self.update_config()
                    # 按文件序号
                    elif os.path.exists('./' + self.conf["folder"] + self.all_image_names[int(arg[1])]):
                        self.conf['image'] = self.all_image_names[int(arg[1])]  # 装载
                        self.updatePrompt()  # 更新指示符
                        self.updateItemList()  # 更新项目列表
                        self.update_config()
                    else:
                        print("\033[1;31m" + time.strftime(
                            "%Y-%m-%d %H:%M:%S ") + "ERROR: image file does not exist" + "\033[0m")
                        return 0
                except:
                    print("\033[1;31m" + time.strftime(
                        "%Y-%m-%d %H:%M:%S ") + "ERROR: An unknown error occurred while selecting a image" + "\033[0m")
        # 指令错误
        elif len(arg) > 2:  # 文件或文件夹名有空格
            print("\033[1;31m" + time.strftime(
                "%Y-%m-%d %H:%M:%S ") + "ERROR: File or folder names should not have spaces" + "\033[0m")
            return 0
        elif len(arg) < 2:  # 指令缺少参数
            print("\033[1;31m" + time.strftime(
                "%Y-%m-%d %H:%M:%S ") + "ERROR: not enough parameters" + "\033[0m")
            return 0

    def do_exit(self, _):
        'Exit solaraneldefectdetect v1.0'
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: Exiting")
        sys.exit()

    def do_detect(self, arg):
        'Detection command: start detection after setting pictures and models'
        debug = False
        if arg == "debug":
            debug = True
        if self.conf['image'] != "":  # 图片存在且已加载
            start_time = time.time()
            image_path = "./" + self.conf['folder'] + self.conf['image']
            print("\033[1;32m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "DETECT: reading" + "\033[0m")
            image = cv2.imread(image_path)  # 原图
            print("\033[1;32m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "DETECT: correcting" + "\033[0m")
            image_cor = correct(image, debug=debug)  # 校正图
            draw_image = image_cor.copy()
            print("\033[1;32m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "DETECT: segmenting" + "\033[0m")
            seg_list = segment(image_cor, seg_method=4, debug=debug)  # 坐标，分割图
            slices_num = len(seg_list)  # 切片总数
            damaged_num = 0  # 缺陷切片数
            with alive_bar(slices_num) as bar:
                for item in seg_list:  # 遍历任务
                    bar.text = "\033[1;32m" + "-> DETECT: detecting, please wait..." + "\033[0m"
                    bar()  # 显示进度
                    point = item[0]
                    img = item[1]
                    img = self.segArrayPreprocess(img, self.conf['model'])  # 预处理
                    result = self.Model(img)  # 送入模型
                    conclusion = self.judgment(result, self.conf['model'])  # 统一判断数
                    if conclusion == 0:  # 检测到缺陷
                        damaged_num += 1  # 缺陷数+1
                        # draw_image = cv2.rectangle(draw_image, tuple(point[0]), tuple(point[1]), (255, 0, 0), 10)     # 框选缺陷
                        draw_image = cv2.drawMarker(draw_image, tuple(((point[0] + point[1]) / 2).astype(np.int32)),
                                                    (255, 0, 0), markerType=1, markerSize=100, thickness=10)  # 缺陷画叉
                    """
                    代码
                    """
            print("\033[1;32m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "DETECT: completed Time: {:.3f}s".format(
                time.time() - start_time) + "\033[0m")
            print("---------------------------------------------------------------------------------------------------")
            print("Detect report:")
            print("File name: " + self.conf['image'])
            print("slices_num: {}\n".format(slices_num) + "\033[1;32m" +
                  "perfect_num: {}\n".format(slices_num - damaged_num) + "\033[0m" + "\033[1;31m" +
                  "damaged_num: {}\ndamaged_rate: {:.3f}%".format(damaged_num,
                                                                  damaged_num / slices_num * 100) + "\033[0m")
            print("---------------------------------------------------------------------------------------------------")
            plt.subplot(121)
            plt.xlabel("Corrected")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image_cor)
            plt.subplot(122)
            plt.xlabel("Detected")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(draw_image)
            plt.show()

    def do_about(self, arg):
        'Show author information'
        print("       =[ \033[1;33msolarpaneldefectdetect v1.0\033[0m ]")
        print("+ -- --=[ Author:                     ]")
        print("+ -- --=[        LiuJue      211804   ]")
        print("+ -- --=[        ZhangEnJin  211920   ]")
        print("+ -- --=[        ShaoJiaWei  211882   ]")
        print("+ -- --=[        ZhangYiChi  211613   ]")

    def default(self, line):
        print(
            "\033[1;33m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "WARNING: Unknown command" + "\033[0m")

    def emptyline(self):
        print(
            "\033[1;33m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "WARNING: Command cannot be empty" + "\033[0m")

    def preloop(self):
        intro = "   _____         __              ____                       __ ____         ____             __   ____         __               __ \n" \
                "  / ___/ ____   / /____ _ _____ / __ \ ____ _ ____   ___   / // __ \ ___   / __/___   _____ / /_ / __ \ ___   / /_ ___   _____ / /_\n" \
                "  \__ \ / __ \ / // __ `// ___// /_/ // __ `// __ \ / _ \ / // / / // _ \ / /_ / _ \ / ___// __// / / // _ \ / __// _ \ / ___// __/\n" \
                " ___/ // /_/ // // /_/ // /   / ____// /_/ // / / //  __// // /_/ //  __// __//  __// /__ / /_ / /_/ //  __// /_ /  __// /__ / /_  \n" \
                "/____/ \____//_/ \__,_//_/   /_/     \__,_//_/ /_/ \___//_//_____/ \___//_/   \___/ \___/ \__//_____/ \___/ \__/ \___/ \___/ \__/  \n" \
                "                                                                                                                                   \n" \
                "       =[ \033[1;33msolarpaneldefectdetect v1.0\033[0m ]\n" \
                "+ -- --=[ step1: select model         ]\n" \
                "+ -- --=[ step2: select image         ]\n" \
                "+ -- --=[ step3: detect!              ]\n"
        print(intro)  # 打印介绍信息

        self.read_config()  # 读取config.json 存储到self.conf
        ######## 读取配置后继续 ##########
        if self.conf['model'] == 'SVM':  # SVM
            self.Model = tf.keras.models.load_model(SVM_path)
            self.updatePrompt()  # 更改指示符
        elif self.conf['model'] == 'DenseNet':  # DenseNet
            self.Model = tf.keras.models.load_model(DenseNet_path)
            self.updatePrompt()  # 更改指示符
        self.updateItemList()  # 更新项目列表
        self.check_images()  # 如果文件夹没有图片，打印警告信息

    def read_config(self, conf_path="config.json"):
        default_conf = {"folder": "./", "model": 'SVM', "image": ""}  # 默认设置
        try:  # 尝试读取设置
            with open(conf_path, "r") as f:
                conf_dict = json.load(f)
                self.conf['folder'] = conf_dict['folder']
                self.conf['model'] = conf_dict['model']
                self.conf['image'] = conf_dict['image']
        except FileNotFoundError:  # 找不到配置文件
            print(time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: No configuration file found")
            with open(conf_path, "w") as f:
                json.dump(default_conf, f)
                print(time.strftime("%Y-%m-%d %H:%M:%S ") + "INFO: Created configuration to" + conf_path)
            with open(conf_path, "r") as f:
                conf_dict = json.load(f)
                self.conf['folder'] = conf_dict['folder']
                self.conf['model'] = conf_dict['model']
                self.conf['image'] = conf_dict['image']
        except KeyError:  # 配置格式错误
            print(
                "\033[1;33m" + time.strftime("%Y-%m-%d %H:%M:%S ") + "WARNING: Configuration format error" + "\033[0m")
            with open(conf_path, "w") as f:
                json.dump(default_conf, f)
                print("\033[1;33m" + time.strftime(
                    "%Y-%m-%d %H:%M:%S ") + "WARNING: Reset to default configuration" + "\033[0m")
            with open(conf_path, "r") as f:
                conf_dict = json.load(f)
                self.conf['folder'] = conf_dict['folder']
                self.conf['model'] = conf_dict['model']
                self.conf['image'] = conf_dict['image']
        except:  # 未知错误
            print("\033[1;31m" + time.strftime(
                "%Y-%m-%d %H:%M:%S ") + "ERROR: An unknown error occurred while reading the configuration file" + "\033[0m")
            sys.exit()  # 强制退出

    def update_config(self, conf_path="config.json"):
        with open(conf_path, "w") as f:
            json.dump(self.conf, f)

    def check_images(self):
        if len(self.all_image_names) == 0:
            print("\033[1;33mWARNING: no pictures available in the current folder\033[0m")
        else:
            print("INFO: There are {} images available under this folder".format(len(self.all_image_names)))

    def updateItemList(self):
        data_root = pathlib.Path('./')  # 设置数据根目录
        self.all_folder_names = sorted(item.name for item in data_root.glob('./*') if item.is_dir())  # 获取文件夹名
        self.all_image_names = sorted(
            item.name for item in data_root.glob('./' + self.conf['folder'] + '*.JPG'))  # 获取文件夹名

    def updatePrompt(self):
        self.prompt = 'detecter ' + self.conf['model'] + '(\033[1;31m' + self.conf['folder'] + self.conf[
            'image'] + '\033[0m)' + ' >'  # 更改指示符

    def segArrayPreprocess(self, image: np.ndarray, model='SVM'):
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

    def judgment(self, result, model='SVM'):
        if model == 'SVM':
            if result < 0:  # SVM负数为缺陷
                return 0
            else:
                return 1  # SVM正数为完好
        elif model == 'DenseNet':
            return int(tf.argmax(result, axis=1))


if __name__ == '__main__':
    try:
        os.system('cls')
        console = DetectConsole()
        console.cmdloop()
    except:
        exit()
