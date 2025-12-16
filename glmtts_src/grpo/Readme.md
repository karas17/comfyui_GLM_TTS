# GLMTTS-GRPO: Multi-Reward Reinforcement Learning

You can finetune the GLMTTS LLM using multiple reward functions with the DeepSpeed framework.

## Environment Setup
Install the required libraries and dependencies according to you need.

**Similarity Reward**
```bash
cd grpo/modules
git clone https://github.com/s3prl/s3prl
```
Download [wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view), and place it in ```grpo/ckpt```.

**Laughter Reward**
```bash
cd grpo/modules
git clone https://github.com/omine-me/LaughterSegmentation
```

Refer to each repository’s README for specific installation and setup instructions, including package requirements and environment configuration.

## Data Preparation
You can uncomment the relevant lines in ```jsonl_generate``` function of [glmtts_inference.py](glmtts_inference.py) to save prompt features.

Run [grpo/prepare_data.py](grpo/prepare_data.py) to exports JSONL files as [grpo/data/sample.jsonl](grpo/data/sample.jsonl).

The configuration file [grpo/data/sample.yaml](grpo/data/sample.yaml) pecifies the JSONL files to be used as training input.

## Reward Server
Select the required reward functions in [grpo/reward_server.py](grpo/reward_server.py).

Start the server on your machine with multiple GPUs:

```bash
cd grpo
bash run_server.sh
# 8 servers listenon ports 8080 to 8087 on GPU 0~4
```
Ensure communication between your server and training nodes.

## Training
Run [grpo/train_ds_grpo.py](grpo/train_ds_grpo.py) with the DeepSpeed engine.

To run a training script on a single GPU:
```bash
bash grpo/pretrain_GRPO_single.sh
# This script runs pretrain-GRPO on GPU 1
```

## Acknowledgement
This work was inspired by the implementation in [GRPO-Zero][1].

[1]: https://github.com/policy-gradient/GRPO-Zero
