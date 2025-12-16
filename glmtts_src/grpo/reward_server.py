# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import zhconv
from funasr import AutoModel
import argparse
import sys
from run_wer import process_one
from run_sim import get_sim_model
from run_sim import verification2
from run_laughter import build_model, recog_laughter

from fastapi import FastAPI
from pydantic import BaseModel
import torchaudio
import torch
import os
import numpy as np
import struct
import collections
import mmap
import pickle
from pathlib import Path
import math
import re

app = FastAPI()
sample_rate = 24000

# 获取worker序号（uvicorn/gunicorn）
worker_id = int(os.environ.get('WORKER_ID', 0)) if 'WORKER_ID' in os.environ else (os.getpid() % 8)
# 或在启动命令里用UVICORN_WORKER=0~7做专门分配

NUM_GPUS = 4
local_rank = worker_id % NUM_GPUS 

def has_consecutive_laugh(text):
    # 匹配由“哈”或“嘿”或“呵”组成的连续2次或以上
    pattern = r'(哈{2,}|嘿{2,}|呵{2,})'
    return re.search(pattern, text) is not None

LAUGHTER_SET = {'呵', '哈', '嘿'}

def check_laughter_list(lst):
    # lst 必须非空，且全部在 LAUGHTER_SET
    return bool(lst) and set(lst).issubset(LAUGHTER_SET)

def cal_token_level_cer(timestamps, alignment, hypo, truth):
    token_count = math.ceil(timestamps[-1][1] / 1000 * 25)
    reward = np.ones(token_count, dtype=np.int32)

    max_len = len(timestamps)
    laugher_deletion = False

    for chunk in alignment:
        if chunk.type == 'equal':
            continue

        # Laugh Deletion 检查
        if chunk.type == "delete" and check_laughter_list(truth[chunk.ref_start_idx:chunk.ref_end_idx]):
            print('encounter laughter words delete')
            laugher_deletion = True
            continue

        # 计算区间
        # start_idx = max(chunk.hyp_start_idx - 1, 0)
        # end_idx = min(chunk.hyp_end_idx + 1, max_len - 1)
        
        if chunk.type == "delete":
            start_idx = max(chunk.hyp_start_idx-1, 0)
            end_idx = min(chunk.hyp_end_idx, max_len-1)
            start_time = timestamps[start_idx][1]
            end_time = timestamps[end_idx][0]
        elif chunk.type == 'substitute' or chunk.type == 'insert':
            start_idx = max(chunk.hyp_start_idx, 0)
            end_idx = min(chunk.hyp_end_idx-1, max_len-1)
            start_time = timestamps[start_idx][0]
            end_time = timestamps[end_idx][1]

        # 边界异常处理
        if start_idx >= len(timestamps) or end_idx >= len(timestamps):
            print(f"Index error in timestamp: start {start_idx}, end {end_idx}, length {len(timestamps)}")
            continue

        # start_time = timestamps[start_idx][0]
        # end_time = timestamps[end_idx][1]
        # start_time -= 0.03
        # end_time -= 0.03

        start_token = int(start_time / 1000 * 25)
        end_token = int(math.ceil(end_time / 1000 * 25))

        reward[start_token:end_token] = 0

    return reward.tolist(), laugher_deletion

from threading import Lock

sim_lock = Lock()
emo_lock = Lock()
asr_lock = Lock()
laugh_lock = Lock()

class RewardRequest(BaseModel):
    audio_path: str
    uttid: str
    target_audio: str
    ref_text: str
    emotion: int

@app.post("/reward")
def get_reward(req: RewardRequest):
    audio_path = req.audio_path
    uttid = req.uttid
    # target_audio = torch.tensor(req.target_audio, dtype=torch.float32)
    target_audio_path = req.target_audio
    ref_text = req.ref_text
    emotion = req.emotion

    sim_reward = 0
    cer_reward = 0
    nll_reward = 0
    emo_reward = 0
    emo_neg_reward = 0
    token_level_cer_reward = [0]
    laughter_reward = 0
    laughter_deletion = True
    
    try:
    # 1. 读取audio
    # print(audio_path, target_audio)
        response_audio, file_sr = torchaudio.load(audio_path)
        if file_sr != sample_rate:
            response_audio = torchaudio.functional.resample(response_audio, file_sr, sample_rate)
        
        if target_audio_path and target_audio_path != 'None' and sim_model is not None:
            target_audio, file_sr = torchaudio.load(target_audio_path)
            if file_sr != sample_rate:
                target_audio = torchaudio.functional.resample(target_audio, file_sr, sample_rate)
                if target_audio.shape[0] > 1:
                    target_audio = target_audio.mean(dim=0, keepdim=True)
            # print(response_audio.shape, target_audio.shape)
            # 2. sim_model
            with sim_lock:
                sim = verification2(response_audio, target_audio, sim_model).item()
                # sim = 0
                sim_reward = (1+sim)/2
        else:
            sim_reward = 0

        # 3. emo_model
        if emo_model is not None:
            with emo_lock:
                res = emo_model.generate(audio_path, output_dir="./tmp", granularity="utterance", extract_embedding=False)
                if emotion == -1: # emotion对应的tag
                    emo_reward = 1 - res[0]['scores'][4]
                else:
                    emo_reward = res[0]['scores'][emotion]
                emo_neg_reward = - res[0]['scores'][4]

        # 4. asr_model
        if asr_model is not None:
            with asr_lock:
                asr_res = asr_model.generate(input=audio_path, batch_size_s=300)
                transcription = asr_res[0]["text"]
                timestamps = asr_res[0]["timestamp"]
                transcription = zhconv.convert(transcription, 'zh-cn')
                _, _, wer, truth, hypo, alignment = process_one(transcription, ref_text, 'zh')
                cer_reward = float(np.exp(-2.5*wer))
                # token_level_cer_reward, laughter_deletion = cal_token_level_cer(timestamps, alignment, hypo, truth)

                # nll = asr_res[0]['score']
                # nll_reward = np.exp(nll/3)
        # laugh
        if laugh_model is not None:
            with laugh_lock:
                laughters = recog_laughter(audio_path, laugh_model, './tmp')
                # {'0': {'start_sec': 14.461832061068701, 'end_sec': 14.90267175572519}}
                if len(laughters) > 0 and has_consecutive_laugh(ref_text):
                    if laughter_deletion:
                        laughter_reward = 1
                    else:
                        laughter_reward = 0
                else:
                    laughter_reward = 0 
    except Exception as e:
        print("reward error: ", e)
        print(audio_path, target_audio_path)
        print(ref_text, emotion)
        # torchaudio.save()

    return {
        "reward": cer_reward + sim_reward + emo_reward,
        "reward_info": {
            "sim_reward": sim_reward,
            "cer_reward": cer_reward,
            "nll_reward": nll_reward,
            "emo_reward": emo_reward,
            "emo_neg_reward": emo_neg_reward,
            "token_cer_reward": token_level_cer_reward,
            "laughter_reward": laughter_reward,
        }
    }
    
# def get_args():
#     parser = argparse.ArgumentParser(description='training your network')
#     parser.add_argument("--reward_func", type=str, default="emo,cer,sim")
#     args = parser.parse_args()
#     return args

args = {'reward_func': "emo,cer,sim,laugh"}

if 'cer' in args['reward_func']:
    asr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
                    disable_pbar=True, 
                    disable_update=True,
                    device=f"cuda:{local_rank}")
else:
    asr_model = None
    
if 'sim' in args['reward_func']:
    sim_model = get_sim_model(torch.device("cuda", local_rank))
else:
    sim_model = None
    
if 'emo' in args['reward_func']:
    emo_model = AutoModel(model="iic/emotion2vec_plus_large", disable_pbar=True, device=f"cuda:{local_rank}")
else:
    emo_model = None
    
if 'laugh' in args['reward_func']:
    laugh_model = build_model('modules/LaughterSegmentation/models/model.safetensors', local_rank, sample_rate)
else:
    laugh_model = None