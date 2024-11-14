# Video-based-Action-Recognition

## 项目结构

下面简要介绍了一些本项目的项目结构

```
Video-based-Action-Recognition/
│
├── README.md
│
├── myvediodataset/  该文件夹用来放模型训练所需的数据，其中的子文件夹名称就是标签
│   ├── openfile1/
│   └── openmail1/
│
├── Output/           测试数据输出目录
│   └── output.mp4
│
├── test_videos/      用来存放模型推理时用的测试数据
│   └── 20241025_105625.mp4
│
├── LSTM_model_2024_11_01_14_04_18.h5 训练之后的模型输出
│
├── lstm_model_inference.py           模型推理代码
│
├── lstm_model_inference.sh           模型推理调用脚本
│
├── lstm_model_train.py               模型训练代码
│
├── lstm_model_train.sh               模型训练脚本
│
└──  lstm_model.py                    基于桌面端操作视频的动作识别模型类

```

## 环境配置

略

## 数据集准备

这里的数据集实际上时按照动作类别分类了的短视频的集合，每类动作对应一个目录，每个目录下有很多视频文件，在制作自己的数据集时，可以像这样把项目应用的视频数据按动作分类这样分目录存放，一个目录对应一个动作。

### 准备你的训练视频

对于每一个训练视频，这里建议视频里只出现你想预定义好的动作，视频长度大概在 10 秒左右。每个类别的训练视频尽可能越多越好，越丰富越好（这里的丰富指的是视频的内容，比如同一类别下，不同的视频虽然都是在描述同一个动作，但是视频的差别较大：（比如操作终端的背景不同）。尽量避免出现单一化的视频集，让数据更泛化，这样可以提升模型在不同场景下的可用性。

注意这里的动作类别，在后面模型训练时还需要用到，会作为一个参数传进模型训练代码中去，比如这里的 myvdiodataset 下有两个子文件夹：openfile1 openmail1 ,在后续运行` lstm_model_train.py`时就需要传入参数 `--classes_list openfile1 openmail1 `

## 启动这个项目

### 模型训练

直接在命令行中运行下列命令即可

```
bash lstm_model_train.sh
```

注意，`lstm_model_train.sh`中会设置若干参数，它们的含义如下：

```
--dataset_dir ：

指定用于训练模型的数据集所在的目录路径。
--classes_list：

列出数据集中包含的类别名称。
--image_height ：

设置输入图像resize后的高度。
--image_width ：

设置输入图像resize后的宽度。
--sequence_length ：

设置输入序列的长度，这对基于时间序列的模型（如 LSTM）是重要的，因为它们需要一定数量的连续帧来进行预测。
--seed ：

设置随机种子
--epochs ：

设置训练过程中的迭代次数（epoch）。
--batch_size ：

设置每次迭代中用于训练模型的样本数量（batch size）
```

### 模型推理

直接在命令行中运行下列命令即可

```

bash lstm_model_inference.sh

```

注意，`lstm_model_inference.sh`中会设置若干参数，它们的含义如下：

```

--model_path ：

指定预训练 LSTM 模型文件的路径。这个 .h5 文件包含了模型的权重和结构，用于对视频数据进行预测。
--input_video_file_path ：

指定要进行预测的视频文件的路径。这个视频将被输入到模型中，以识别其中的动作或事件。
--classes_list ：

列出模型可以预测的类别名称。

```
