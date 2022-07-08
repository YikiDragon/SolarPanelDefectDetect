# 光伏电池片图像缺陷检测器

本缺陷检测器针对倾斜的光伏电池板组件照片，应用直方图自适应二值化和透视变换技术进行图像校正，提取行列特征后通过FFT频谱分析出晶片的行列排布进行图像分割，可分别应用非线性SVM与DenseNet对分割照片进行训练以实现缺陷检测。


# 用法

该检测器代码采用python编写，务必保证环境依赖：
- python3.8
- tensorflow: >=2.0.0 & <=2.3.0
- python-opencv: ==4.5.1
- numpy
- matplotlib
- alive_progress
- cmd


## 获取代码

可以以下通过git指令获取代码：  
`git clone https://github.com/YikiDragon/SolarPanelDefectDetect.git`

## 环境依赖

可以通过输入以下指令以满足最基本配置：  
`conda install tensorflow==2.3.0 opencv==4.5.1 numpy matplotlib alive_progress cmd`  
或者在获取代码后在代码根目录输入：  
`conda env create -f require.yaml`

## 目录配置
在获取代码后，在根目录下新建文件夹`photos`用于存放待校正分割的光伏电池板原图。  
## 快速使用
1. 在项目根目录下运行以下指令进入命令交互界面  
`python main.py`  
2. 查看可用文件夹列表  
`show folder`  
3. 选择`photos`文件夹  
`set folder photos`或`set folder <文件夹对应编号>`
4. 查看可用图片文件  
`show image`
5. 选择***.jpg图片  
`set image ***.jpg`或`set image <图片对应编号>`
6. 查看可用缺陷识别模型  
`show model`
7. 选择非线性SVM模型  
`set model SVM`或`set model <模型对应编号>`
8. 开始检测  
`detect`
## 指令说明
有以下指令可用：
`show`: 显示可用选项  
`set`: 设置指定选项  
`detect`: 开始检测  
`help`: 帮助  
`about`: 作者信息  
`exit`: 退出交互式命令行
可以使用`help <指令>`获取指令的相应用法
## 文件目录解释
下图列出本项目的核心结构：  
.  
│  automove_file.py	可执行，根据FinalLabel.xls将./dataset/all/下的图片分配  
│  config.json 命令交互系统的配置文件  
│  FinalLabel.xls 标签表  
│  image_utils.py 可执行可调用，包含图像校正和图像分割程序  
│  LabelList.xls   
│  label_convert.py 可执行，根据dataset/下perfect和damaged内文件生成标签表  
│  main.py 可执行，命令交互系统入口  
│  require.yaml conda依赖列表  
│  
├─photos 存放未校正未分割原图的图片  
│  
├─dataset 数据集文件夹  
│  │  
│  ├─train 存放训练用的已校正分割的图片  
│       │  
│       ├─perfect 完好集文件夹  
│       └─damaged 缺陷集文件夹  
│  
├─DenseNet 密集连接网络文件夹  
│  │  Demo.py 可执行，演示文件  
│  │  DenseNet.py 不可执行可调用，网络基本结构文件  
│  │  model.py 不可执行可调用，密集连接网络结构文件  
│  │  test.py 可执行，测试文件  
│  │  train.py 可执行，训练文件  
│  │  utils.py 可执行，图片预处理  
│  │  
│  ├─saved_model 保存的模型  
│  
└─SVM_Kernel 非线性SVM文件夹  
│  │  Demo.py 可执行，演示文件  
│  │  KernelSVM_model.py 不可执行可调用，非线性SVM结构文件  
│  │  test.py 可执行，测试文件  
│  │  train.py 可执行，训练文件  
│  │  utils.py 可执行，图片预处理  
│  │  
│  └─saved_model 保存的模型  