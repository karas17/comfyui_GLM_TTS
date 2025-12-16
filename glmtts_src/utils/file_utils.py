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
import torchaudio
import json
import os

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.to("cuda")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sr
                ).to('cuda')
        speech = resampler(speech).to("cuda")
    return speech

def get_jsonl(jsonl_file_path=None):
    results = []
    lines = open(jsonl_file_path, encoding='utf-8').readlines()
    for arr in lines:
        arr = json.loads(arr)
        uttid, prompt_text, prompt_speech, syn_text = arr["uttid"], arr["prompt_text"], arr["prompt_speech"], arr["syn_text"]
        data = {
            "uttid": uttid,
            "prompt_text": prompt_text,
            "prompt_speech": prompt_speech,
            "syn_text": syn_text,
        }
        results.append(data)

    return results