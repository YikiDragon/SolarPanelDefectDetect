# 光伏电池片图像缺陷检测器

本缺陷检测器针对倾斜的光伏电池板组件照片，应用直方图自适应二值化和透视变换技术进行图像校正，提取行列特征后通过FFT频谱分析出晶片的行列排布进行图像分割，可分别应用非线性SVM与DenseNet对分割照片进行训练以实现缺陷检测。

## 获取代码

可以以下通过git指令获取代码：  

```
git clone https://github.com/YikiDragon/SolarPanelDefectDetect.git
```

## 环境依赖

该检测器代码采用python编写，务必保证环境依赖：

- python3.8
- tensorflow: >=2.0.0 & <=2.3.0
- python-opencv: ==4.5.1
- numpy
- matplotlib
- alive_progress
- cmd
- xlrd
- xlwt

可以通过输入以下指令以满足最基本配置：  

```
conda install tensorflow==2.3.0 opencv==4.5.1 numpy matplotlib alive_progress cmd xlrd xlwt
```  

或者在获取代码后在代码根目录输入：  

```
conda env create -f require.yaml
```

## 快速使用

### 目录配置

在获取代码后，在根目录下新建文件夹`photos`  

```
mkdir photos
```  

在`photos`文件夹中存放未校正分割的光伏电池板原图。
>光伏电池板原图须为".jpg"或".JPG"  

### 使用交互式命令行检测

1. 在项目根目录下运行以下指令启动交互式命令行

```
python main.py
```  

2. 查看可用文件夹列表  

```
show folder
```  

3. 选择`photos`文件夹  

```
set folder photos
```

>或`set folder <文件夹对应编号>`  

4. 查看可用图片文件  

```
show image
```

5. 选择***.jpg图片  

```
set image ***.jpg
```

>或`set image <图片对应编号>`  

6. 查看可用缺陷识别模型  

```
show model
```

7. 选择非线性SVM模型  

```
set model SVM
```

>或`set model <模型对应编号>`  

8. 开始检测  

```
detect
```

### 指令说明

有以下指令可用：  
`show`: 显示可用选项  
`set`: 设置指定选项  
`detect`: 开始检测  
`help`: 帮助  
`about`: 作者信息  
`exit`: 退出交互式命令行  
>可以使用`help <指令>`获取指令的相应用法

## 详细用法

### 自动化校正分割

1. 在根目录下新建数据集文件夹`dataset`及其子文件夹  

```
mkdir dataset  
cd dataset
mkdir all           # 存放已分割未打标图片的文件夹
cd ..
```

2. 在`photos`文件夹中已有原图的情况下运行`autosegment.py`，所有原图将被自动分割并保存至`./dataset/all`

```
python autosegment.py
```

### 自动化分配标签

1. 打开`dataset`文件夹建立训练集文件夹

```
cd dataset
mkdir train         # 存放训练集的文件夹
cd train
mkdir perfect       # 完好集
mkdir damaged       # 缺陷集
cd ../..
```

2. 在`dataset/all`中已有校正分割图的情况下运行`automove.py`，所有分割图将按照公共标签表`FinalLabel.xls`移动至`perfect`或`damaged`文件夹作为打标

```
python automove.py
```

### 根据自定义标签自动生成标签表

如果您已经手动将校正分割后的图片分配到`perfect`和`damaged`，可以运行`label_convert.py`将您的分配结果编写为Excel表格

```
python label_convert.py
```

生成的自定义标签表在`LabelList.xls`

### 模型训练

在`dataset/train/perfect`和`dataset/train/damaged`不为空的情况下，可以进行模型训练。

1. 进入检测模型文件夹例如:

```
cd DenseNet
```

2. 运行`train.py`开始训练

```
python train.py
```

>一些详细的配置(例如: 优化器, BatchSize, Iterations, Epochs等)可以在`train.py`中修改

3. 生成Precision-Recall曲线并计算AP

```
python test.py
```

4. 随机测试分割图识别效果

```
python Demo.py
```

### 关键函数调用方法

1. 图像校正

```
from image_utils.py import correct
image_corrected = correct(img_src, debug=False)     # debug用于输出一些中间过程的分析图表
```

>可参考`autosegment.py`

2. 图像分割  

```
from image_utils.py import segment
seg = segment(image_corrected, seg_method=4, debug=False)   # 务必选择稳定性最强的第4个分割方法——频谱分析法
```

>可参考`autosegment.py`

3. 缺陷识别

```
import tensorflow as tf
model = tf.keras.models.load_model(model_path)  # 加载模型
pred = model(x)                                 # 模型预测,x为预处理后的图片或特征向量
```

>可参考模型文件夹下的`Demo.py`，模型存储位置为模型文件夹下的`saved_model`

## 文件目录解释

下图列出本项目的核心结构：  
.  
│  automove.py可执行，根据FinalLabel.xls将./dataset/all/下的图片分配到perfect或damaged  
│  autosegment.py 可执行，自动将photos中的图片分割到./dataset/all/  
│  config.json 命令交互系统的配置文件  
│  FinalLabel.xls 公共标签表  
│  image_utils.py 可执行可调用，包含图像校正和图像分割程序  
│  LabelList.xls  自定义标签表  
│  label_convert.py 可执行，根据dataset/下perfect和damaged内文件生成标签表  
│  main.py 可执行，命令交互系统入口  
│  require.yaml conda依赖列表  
│  
├─photos 存放未校正未分割原图的图片  
│  
├─dataset 数据集文件夹  
│  │  
│  ├─train 存放训练用的已打标的图片  
│  │    │  
│  │    ├─perfect 完好集文件夹  
│  │    └─damaged 缺陷集文件夹  
│  │  
│  └─all 存放已校正分割未打标的图片  
│  
├─DenseNet 密集连接网络文件夹  
│  │  Demo.py 可执行，演示文件  
│  │  DenseNet.py 不可执行可调用，网络基本结构文件  
│  │  model.py 不可执行可调用，密集连接网络结构文件  
│  │  test.py 可执行，测试文件  
│  │  train.py 可执行，训练文件  
│  │  utils.py 可执行，图片预处理  
│  │  
│  └─saved_model 保存的模型  
│  
└─SVM_Kernel 非线性SVM文件夹  
│  │  Demo.py 可执行，演示文件  
│  │  KernelSVM_model.py 不可执行可调用，非线性SVM结构文件  
│  │  test.py 可执行，测试文件  
│  │  train.py 可执行，训练文件  
│  │  utils.py 可执行，图片预处理  
│  │  
│  └─saved_model 保存的模型
