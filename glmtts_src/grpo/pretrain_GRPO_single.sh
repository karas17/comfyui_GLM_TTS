#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES="1"

# 去往项目根目录
root_dir=$(dirname $(dirname "$(readlink -f "$0")"))
#root_dir=/your/project/
echo "current script root dir: $root_dir"
cd $root_dir
mkdir $root_dir/grpo/temp_samples

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PROJECT=$root_dir
export PYTHONPATH=$PYTHONPATH:$PROJECT

text_tokenizer="default"
base_nm=$(basename $0)
exp_nm="${base_nm%.*}"

name="pretrain-grpo-test"
python_cmd=" \
grpo.train_ds_grpo \
--multinode=False \
--train_engine=deepspeed \
--name=$name \
--tensorboard_dir=${root_dir}/tensorboard/${name} \
--data-patterns=${root_dir}/grpo/data/sample.yaml \
--config=${root_dir}/grpo/config/lm_llama_casual_glm_32k_GRPO_cer.yaml \
--text_tokenizer=$text_tokenizer \
--seed=10086 \
--save_root=${root_dir}/ckpt \
--accumulate_gradient=1 \
--deepspeed_config=${root_dir}/grpo/config/deepspeed.json \
--deepspeed.save_states=model+optimizer \
--mode=PRETRAIN \
--dryrun=False \
--flow=pretrain_flow \
"

mpi_cmd="python -m ${python_cmd}"
eval ${mpi_cmd} 2>&1 | tee logs/${MLP_TASK_ID}/output.log