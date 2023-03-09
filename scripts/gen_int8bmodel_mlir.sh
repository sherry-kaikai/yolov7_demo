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
        --model_def ../models/onnx/yolov7_v0.1_3output_$1b.onnx \
        --input_shapes [[$1,3,640,640]] \
        --mlir yolov7_v0.1_3output_$1b.mlir \
        --mean 0.0,0.0,0.0 \
        --scale 0.0039216,0.0039216,0.0039216 \
        --keep_aspect_ratio \
        --pixel_format rgb
}

function gen_cali_table()
{
    run_calibration.py yolov7_v0.1_3output_$1b.mlir \
        --dataset ../datasets/coco128 \
        --input_num 128 \
        -o yolov7_cali_table
}

function gen_int8bmodel()         # --asymmetric \
{
    model_deploy.py \
        --mlir yolov7_v0.1_3output_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --quantize_table yolov7_qtable \
        --calibration_table yolov7_cali_table \
        --model yolov7_v0.1_3output_int8_$1b.bmodel

    mv yolov7_v0.1_3output_int8_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_mlir 1
# if [ $? -eq 0 ]; then
#     echo "Congratulation! step0: mlir 1 batch is done!"
# else
#     echo "step0 Something is wrong, pleae have a check!"
#     popd
#     exit -1
# fi
gen_cali_table 1
# if [ $? -eq 0 ]; then
#     echo "Congratulation! step1: mlir 1 batch is done!"
# else
#     echo "step1 Something is wrong, pleae have a check!"
#     popd
#     exit -1
# fi
gen_int8bmodel 1
# if [ $? -eq 0 ]; then
#     echo "Congratulation! mlir 1 batch is done!"
# else
#     echo "gen int8bmodel_1batch Something is wrong, pleae have a check!"
#     popd
#     exit -1
# fi
# batch_size=4
# gen_mlir 4
# if [ $? -eq 0 ]; then
#     echo "Congratulation! step1: mlir 4 batch is done!"
# else
#     echo "Something is wrong, pleae have a check!"
#     popd
#     exit -1
# # fi
# gen_cali_table 4
# gen_int8bmodel 4

popd