#!/bin/bash
scripts_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684"
MODE="pcie_test"
TPUID=0
ALL_PASS=1

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID]" 1>&2 
}

while getopts ":m:t:s:d:" opt
do
  case $opt in 
    m)
      MODE=${OPTARG}
      echo "mode is $MODE";;
    t)
      TARGET=${OPTARG}
      echo "target is $TARGET";;
    s)
      SOCSDK=${OPTARG}
      echo "soc-sdk is $SOCSDK";;
    d)
      TPUID=${OPTARG}
      echo "using tpu $TPUID";;
    ?)
      usage
      exit 1;;
  esac
done

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    ALL_PASS=0
  fi
  sleep 3
}

function eval_cpp()
{
  pushd cpp/yolov7_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov7_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./yolov7_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1"
  echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  echo "python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017.json --result_json results/$3_val2017_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log"
  echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017.json --result_json results/$3_val2017_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  popd
}

function test_python()
{
  python3 python/yolov7_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID
  judge_ret $? "python3 python/yolov7_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID"
}

function eval_python()
{  
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/yolov7_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yolov7_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1"
  echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  echo "python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017.json --result_json results/$2_val2017_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log"
  echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017.json --result_json results/$2_val2017_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
}

 eval_python opencv yolov7_v0.1_3output_fp32_1b.bmodel 0.3728494154948667