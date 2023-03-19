# YOLOv7

## 目录

- [YOLOv7](#yolov7)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备数据与模型](#3-准备数据与模型)
  - [4. 模型编译](#4-模型编译)
    - [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
    - [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)
  



## 1. 简介

​	`YOLOv7`在 5 FPS 到 160 FPS 范围内的速度和准确度都超过了所有已知的目标检测器，并且在 GPU V100 上 30 FPS 或更高的所有已知实时目标检测器中，具有最高的准确度 56.8% AP。

​	`YOLOv7-E6`（56 FPS V100，55.9% AP）比基于`Transformer`的检测器 `SWIN-L Cascade-Mask R-CNN`（9.2 FPS A100，53.9% AP）的速度和准确度分别高出 509% 和 2%，并且比基于卷积的检测器 `ConvNeXt-XL Cascade-Mask R-CNN` (8.6 FPS A100, 55.2% AP) 速度提高 551%，准确率提高 0.7%，以及 `YOLOv7` 的表现还优于：`YOLOR`、`YOLOX`、`Scaled-YOLOv4`、`yolov7`、 `DETR`、`Deformable DETR`、`DINO-7cale-R50`、`ViT-Adapter-B` 和许多其他速度和准确度的目标检测算法。此外，`YOLOv7`基于 `MS COCO` 数据集上从零开始训练 ，未使用任何其他数据集或预训练的权重。

​	本例程对[yolov7官方开源仓库](https://github.com/WongKinYiu/yolov7)v0.1版本的模型喝算法进行移植，使之能在SOPHON BM1684上进行推理测试。


## 2. 特性

* 支持BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和3个输出模型推理
* 支持图片和视频测试

## 3. 准备数据与模型

​	如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型。具体可参考[YOLOv7模型导出](./docs/yolov7_Export_Guide.md)。

​	同时，您需要准备用于测试的数据，如果量化模型，还要准备用于量化的数据集。

​	本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](https://vscode-remote+ssh-002dremote-002b11-002e73-002e12-002e77.vscode-resource.vscode-cdn.net/home/jin.zhang/chencp/examples/LPRNet/README.md#4-模型转换)进行模型转换。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```
下载的模型包括：
```
  
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017                               # coco val2017数据集
    └── instances_val2017.json                # coco val2017数据集标签文件，用于计算精度评价指标  
```


​	模型信息：

| 模型名称 | [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) |
| :------- | :----------------------------------------------------------- |
| 训练集   | MS COCO                                                      |
| 概述     | 80类通用目标检测                                             |
| 输入数据 | images, [batch_size, 3, 640, 640], FP32，NCHW，RGB planar    |
| 输出数据 | [batch_size, 3, 80, 80, 85], FP32 <br />[batch_size, 3, 40, 40, 85], FP32  <br />[batch_size, 3, 20, 20, 85], FP32  <br /> |
| 其他信息 | YOLO_ANCHORS: [12,16, 19,36, 40,28,  36,75, 76,55, 72,146,  142,110, 192,243, 459,401] |
| 前处理   | BGR->RGB、/255.0                                             |
| 后处理   | nms等                                                        |



## 4. 模型编译

​	trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel。

### 4.1 **TPU-NNTC编译BModel**

​	模型编译前需要安装TPU-NNTC，具体方法可参考[《TPU-NNTC开发参考手册》](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在TPU-NNTC环境中进入例程目录。

- 生成FP32 BModelc

使用TPU-NNTC将trace后的torchscript模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETP 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684），如：

```bash
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

执行上述命令会在`models/BM1684/`下生成`yolov7s_v0.1_3output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成INT8 BModel

使用TPU-NNTC量化torchscript模型的方法可参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)，以及[模型量化注意事项](../../docs/Calibration_Guide.md#1-注意事项)。

本例程在`scripts`目录下提供了TPU-NNTC量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台，如：

```shell
./scripts/gen_int8bmodel_nntc.sh BM1684
```

上述脚本会在`models/BM1684`下生成`yolov7_v0.1_3output_int8_1b.bmodel`等文件，即转换好的INT8 BModel。



> **YOLOv7模型量化建议（也可参考[官方量化手册指导](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/calibration-tools/html/module/chapter7.html)）：**
>
> 1. 制作lmdb量化数据集时，通过convert_imageset.py完成数据的预处理；
> 2. 尝试不同的iterations进行量化可能得到较明显的精度提升；
> 3. 最后一层conv到输出之间层之间设置为fp32，可能得到较明显的精度提升；
> 4. 尝试采用不同优化策略，比如：图优化、卷积优化，可能会得到较明显精度提升。

## 5. 例程测试

- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试

### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017)和相关参数(conf_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：

```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov7_v0.1_3output_fp32_1b.bmodel_val2017_opencv_python_result.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：

|   测试平台    |      测试程序     |              测试模型               | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel | 0.514 | 0.699 |
| BM1684 PCIe  | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel | 0.505 | 0.696 |
| BM1684 PCIe  | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel | 0.500 | 0.682 |
| BM1684 PCIe  | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel | 0.495 | 0.682  |
| BM1684 PCIe  | yolov7_bmcv.pcie | yolov7_v0.1_3output_fp32_1b.bmodel | 0.494 | 0.696 |
| BM1684 PCIe  | yolov7_bmcv.pcie | yolov7_v0.1_3output_int8_1b.bmodel | 0.487 | 0.691 |
| BM1684X PCIe | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel | 0.513 | 0.699 |
| BM1684X PCIe | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel | 0.507 | 0.695 |
| BM1684X PCIe | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel | 0.508 | 0.696 |

> **测试说明**：  
1. batch_size=4和batch_size=1的模型精度一致；
2. SoC和PCIe的模型精度一致；
3. AP@IoU=0.5:0.95为area=all对应的指标。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/yolov7_v0.1_3output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/yolov7_v0.1_3output_fp32_1b.bmodel  | 0.083240 |
| BM1684/yolov7_v0.1_3output_fp32_4b.bmodel  | 0.082036 |
| BM1684/yolov7_v0.1_3output_int8_1b.bmodel  | 0.048883 |
| BM1684/yolov7_v0.1_3output_int8_4b.bmodel  | 0.077643 |
| BM1684X/yolov7_v0.1_3output_fp32_1b.bmodel | 0.099467 |
| BM1684X/yolov7_v0.1_3output_fp32_4b.bmodel | 0.09762575 |
| BM1684X/yolov7_v0.1_3output_fp16_1b.bmodel | 0.023757 |
| BM1684X/yolov7_v0.1_3output_fp16_4b.bmodel | 0.022555 |
| BM1684X/yolov7_v0.1_3output_int8_1b.bmodel | 0.009990 |
| BM1684X/yolov7_v0.1_3output_int8_4b.bmodel | 0.009547 |



> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：




|    测试平台  |     测试程序      |             测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| ----------- | ---------------- | ----------------------------------- | -------- | --------- | --------- | --------- |
| BM1684 SoC  | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel | 21.20 | 29.69 | 94.13 | 118.35 |
| BM1684 SoC  | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel | 19.80 | 23.83 | 70.83 | 110.97 |
| BM1684 SoC  | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel | 19.85 | 25.09 | 43.18 | 154.96 |
| BM1684 SoC  | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel | 1.80 | 3.15 | 88.86 | 110.85 |
| BM1684 SoC  | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel | 1.76 | 2.58 | 54.85 | 111.21 |
| BM1684 SoC  | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel | 1.73 | 2.41 | 24.49 | 124.76 |
| BM1684 SoC  | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel | 13.613| 1.920 | 82.888 | 20.474 |
| BM1684 SoC  | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel | 13.687 | 1.895 | 48.742 | 20.055 |
| BM1684 SoC  | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel | 13.469 | 7.349 | 77.204 | 81.145 |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel | 25.45 | 29.60 | 110.80 | 111.46 |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel | 21.82 | 23.23 | 19.83 | 104.05 |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel | 21.81 | 27.29 | 34.96 | 109.69 |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel | 3.76 | 2.41 | 106.03 | 106.09 |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel | 1.71 | 2.38 | 16.44 | 104.05 |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel | 1.67 | 2.41 | 30.33 | 104.40 |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel | 13.18 | 0.845 | 99.40 | 19.355 |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel | 13.18 | 0.845 | 9.851 | 19.852 |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel | 13.147 | 0.847 | 23.644 | 19.320 |


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；
4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异。 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。