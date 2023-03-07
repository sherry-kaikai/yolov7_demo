# YOLOv7模型导出
## 1. 准备工作
YOLOv7模型导出是在Pytorch模型的生产环境下进行的，需提前根据[yolov7官方开源仓库](https://github.com/WongKinYiu/yolov7)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

## 2. 主要步骤
### 2.1 导出torchscript、onnx模型

YOLOv7不同版本的代码导出的YOLOv7模型的输出会有所不同，根据不同的组合可能会有1个、3个输出的情况，主要取决于trace命令参数。Pytorch模型在编译前要经过`torch.jit.trace`，trace后的模型才能使用tpu-nntc编译BModel。YOLOv7官方仓库提供了模型导出脚本`export.py`，可以直接使用它导出torchscript模型：

1输出的trace命令：

```BASH
# 下述脚本可能会根据不用版本的YOLOv7有所调整，请以官方仓库说明为准
python3 export.py --weights yolov7.pt --grid
```

3输出的trace命令：

```BASH
# 下述脚本可能会根据不用版本的YOLOv7有所调整，请以官方仓库说明为准
python3 export.py --weights yolov7.pt
```

上述脚本会在原始pt模型所在目录下生成导出的torchscript、onnx'模型，导出后可以修改模型名称以区分不同版本和输出类型，如`yolov7_v0.1_3output.torchscript.pt` ,`yolov7_v0.1_3output.onnx`表示带有3个输出的JIT模型、onnx模型。

**注意：** 导出的模型建议以`.pt`为后缀，以免在后续模型编译量化中发生错误。

## 3. 常见问题
TODO