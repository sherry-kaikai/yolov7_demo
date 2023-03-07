#!/bin/bash
root_dir=$(dirname $(readlink -f "$0"))/..

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

model_dir="${root_dir}/models/torch"
output_dir="${root_dir}/models/${platform}"

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
echo outdir is $outdir

function auto_cali()
{
    python3 -m ufw.cali.cali_model  \
            --net_name=yolov7  \
            --model=${model_dir}/yolov7_v0.1_3outputs.torchscript.pt  \
            --cali_image_path=./coco128  \
            --cali_iterations=128   \
            --cali_image_preprocess='resize_h=640,resize_w=640;scale=0.003921569,bgr2rgb=True'   \
            --input_shapes="[1,3,640,640]"  \
            --target=$target   \
            --convert_bmodel_cmd_opt="-opt=2"   \
            --try_cali_accuracy_opt="-fpfwd_outputs=< 105 >86,< 105 >55,< 105 >18;-th_method MSE;-conv_group;-per_channel;-accuracy_opt;-graph_transform;-iterations 200;-dump_dist ./calibration_use_pb_dump_dist;-load_dist ./calibration_use_pb_dump_dist" \
            --postprocess_and_calc_score_class=feature_similarity
    mv ${model_dir}/yolov7_batch1/compilation.bmodel $outdir/yolov7_v0.1_3output_int8_1b.bmodel
}

function gen_int8bmodel()
{
    bmnetu --model=../yolov7_bmnetp_deploy_int8_unique_top.prototxt  \
           --weight=../yolov7_bmnetp.int8umodel \
           -net_name=yolov7 \
           --shapes=[$1,3,640,640] \
           -target=$target \
           -opt=1
    mv compilation/compilation.bmodel $outdir/yolov7_v0.1_3output_int8_$1b.bmodel
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
auto_cali
# gen_int8bmodel 1
# batch_size=4
gen_int8bmodel 4

popd