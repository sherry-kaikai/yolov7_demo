#!/bin/bash
model_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

function gen_mlir()
{
    model_transform.py \
        --model_name yolov7_v0.1_3output \
        --model_def ../models/onnx/yolov7_v0.1_3output.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir yolov7_v0.1_3output_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir yolov7_v0.1_3output_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model yolov7_v0.1_3output_fp32_$1b.bmodel

    mv yolov7_v0.1_3output_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
# gen_mlir 1
# gen_fp32bmodel 1

gen_mlir 4
if [ $? -eq 0 ]; then
    echo "Congratulation! step1: mlir 4 batch is done!"
else
    echo "Something is wrong, pleae have a check!"
    popd
    exit -1
fi
gen_fp32bmodel 4
popd