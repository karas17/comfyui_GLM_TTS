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
import sys, os
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import json
import re


punctuation_all = punctuation + string.punctuation



def load_en_model(device):
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model


def load_zh_model():
    model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
                      disable_pbar=True, 
                      disable_update=True)
    return model


def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    wer = compute_measures(truth, hypo)["wer"]
    # ref_list = truth.split(" ")
    # subs = measures["substitutions"] / len(ref_list)
    # dele = measures["deletions"] / len(ref_list)
    # inse = measures["insertions"] / len(ref_list)
    align_res = compute_measures(truth, hypo)
    truth = re.sub(r'\s+', ' ', truth).strip()
    hypo = re.sub(r'\s+', ' ', hypo).strip()
    #print(align_res)
    return (raw_truth, raw_hypo, wer, align_res['truth'][0], align_res['hypothesis'][0], align_res['ops'][0])


