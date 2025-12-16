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
from typing import Any, Dict, List, Optional
import torch
from glmtts_inference import local_flow_forward
import torchaudio
import numpy as np
import zhconv
import os

sample_rate = 24000
test_save = 10
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
import os
import uuid
import librosa

def get_pitch_energy_var(y, sr):
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        np.sum(np.abs(y[i:i+frame_length]**2))
        for i in range(0, len(y)-frame_length, hop_length)
    ])

    # 3. 用librosa计算音高（用pyin或yin算法，需librosa>=0.8.1）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr, 
        frame_length=frame_length, 
        hop_length=hop_length
    )

    # pyin输出的f0中，非发声部分是nan，建议只计算有音高的部分的方差：
    f0_valid = f0[~np.isnan(f0)]
    energy_valid = energy[~np.isnan(f0)]
    energy_var = np.var(energy_valid)
    pitch_var = np.var(f0_valid)
    
    return energy_var, pitch_var

def get_pitch(y, sr):
    frame_length = 2048
    hop_length = 512

    # 3. 用librosa计算音高（用pyin或yin算法，需librosa>=0.8.1）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr, 
        frame_length=frame_length, 
        hop_length=hop_length
    )

    # pyin输出的f0中，非发声部分是nan，建议只计算有音高的部分的方差：
    f0_valid = f0[~np.isnan(f0)]
    pitch_mean = np.mean(f0_valid)
    return pitch_mean

@torch.no_grad()
def reward_function_server(
    uttid: str,
    response_token: List[int],
    prompt_speech_token: torch.Tensor,
    speech_feat: torch.FloatTensor,
    embedding: torch.FloatTensor,
    target_audio: str,
    ref_text: str,
    emotion: torch.Tensor,
    flow,
    server_url="http://172.18.104.111:808"
) -> Dict[str, Any]:
    # 1. 生成 audio 文件并保存
    try:
        response_audio, full_mel = local_flow_forward(flow, response_token, prompt_speech_token, speech_feat, embedding)
        uid = uuid.uuid4()
        save_path = f'{CURRENT_DIR}/temp_samples/{uttid}_{uid}.wav'
        torchaudio.save(save_path, response_audio, sample_rate)
    except Exception as e:
        print("error saving audio:", e)
        return {
            "reward": 0,
            "reward_info": {
                "sim_reward": 0,
                "cer_reward": 0,
                "nll_reward": 0,
                "emo_reward": 0,
                "emo_neg_reward": 0,
                "pitch_reward": 0,
                "energy_reward": 0,
                "token_cer_reward": [0] * len(response_token),
                "laughter_reward": 0,
            }
        }

    # 2. 构造请求
    # if target_audio.shape[0] > 1:
    #     target_audio = target_audio[0].unsqueeze(0)
    data = {
        "audio_path": save_path,
        "uttid": uttid,
        # "target_audio": target_audio.detach().cpu().tolist(),
        "target_audio": target_audio,
        "ref_text": ref_text,
        "emotion": emotion.item(),
    }
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        local_server_url = f'{server_url}{local_rank}/reward'
        resp = requests.post(local_server_url, json=data)
        ret = resp.json()
        
        # try:
        #     # local reward
        #     energy_var, pitch_var = get_pitch_energy_var(response_audio.squeeze(0).cpu().numpy(), sample_rate)
        #     ret["reward_info"]["energy_reward"] = energy_var
        #     ret["reward_info"]["pitch_reward"] = pitch_var
        # except:
        ret["reward_info"]["token_cer_reward"] = (ret["reward_info"]["token_cer_reward"] + [1]*len(response_token))[:len(response_token)]
        ret["reward_info"]["energy_reward"] = 0
        ret["reward_info"]["pitch_reward"] = 0
        if ret["reward_info"]["laughter_reward"] < 1:
            ret["reward_info"]["laughter_reward"] = 0
        
        
    except Exception as e:
        print('reward server error:', e)
        print(local_server_url)
        # print("Status:", resp.status_code)
        # print("Content:", resp.text)
        ret = {
            "reward": 0,
            "reward_info": {
                "sim_reward": 0,
                "cer_reward": 0,
                "nll_reward": 0,
                "emo_reward": 0,
                "emo_neg_reward": 0,
                "pitch_reward": 0,
                "energy_reward": 0,
                "token_cer_reward": [0] * len(response_token),
                "laughter_reward": 0,
            }
        }  
    finally:
        if os.path.exists(save_path):
            # global test_save
            # if ret['reward_info']['laughter_reward'] == 1 and test_save > 0:
            #     test_save -= 1
            # else:
                try:
                    os.remove(save_path)
                except Exception:
                    pass
    return ret

if __name__ == '__main__':
    data_path = 'competitor200/2.wav'
    tgt_audio_path = 'competitor200/9.wav'
    
    target_audio, file_sr = torchaudio.load(tgt_audio_path)
    if file_sr != sample_rate:
        response_audio = torchaudio.functional.resample(target_audio, file_sr, sample_rate)
            
    fake_data = {
        "audio_path": data_path,
        "uttid": '1',
        "target_audio": target_audio.detach().cpu().tolist(),
        "ref_text": '哈哈我也喜欢螺蛳粉，螺蛳粉真的很让人上头'
    }
    server_url="http://localhost:8080/reward"
    resp = requests.post(server_url, json=fake_data)
    ret = resp.json()