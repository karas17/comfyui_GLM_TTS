#!/bin/bash
root_dir=$(dirname "$(readlink -f "$0")")
cd "$root_dir" || exit

get_idle_gpu() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk '{print NR-1, $1 + $2}' | sort -nk2 | head -n1 | cut -d' ' -f1
}

# Configuration
exp_name="_test"
data_list=("example_en" "example_zh")
sample_rate=24000 # 24000 or 32000
args="" # "--use_phoneme" or ""

# Loop through data items and run inference
for data_item in "${data_list[@]}"; do
    idle_gpu=$(get_idle_gpu)
    echo "Launching task: Data=$data_item | GPU=$idle_gpu"
    export CUDA_VISIBLE_DEVICES=$idle_gpu
    python "$root_dir/glmtts_inference.py" \
        --data="$data_item" \
        --exp_name="$exp_name" \
        --sample_rate=$sample_rate \
        $args \
        --use_cache &
    sleep 10
done
wait