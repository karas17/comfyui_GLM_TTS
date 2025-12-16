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
import sys
import os
import sys
import time
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from modules.wavlm_ecapa.ecapa_tdnn import ECAPA_TDNN_SMALL
import librosa

def verification2(wav1, wav2, model):
    device = next(model.parameters()).device
    wav1 = load_wav(wav1).to(device)
    wav2 = load_wav(wav2).to(device)

    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    return sim


def load_wav(wav_arr):
    wav_arr = wav_arr.float()
    wav_arr = resample_func(wav_arr)
    return wav_arr

def get_ckpt():
    ckpt_path = "ckpt/wavlm_large_finetune.pth"
    return ckpt_path

def get_sim_model(device):
    checkpoint = get_ckpt()
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict['model'], strict=False)
    model = model.to(device)
    model.eval()
    return model
resample_func = Resample(orig_freq=24000, new_freq=16000)