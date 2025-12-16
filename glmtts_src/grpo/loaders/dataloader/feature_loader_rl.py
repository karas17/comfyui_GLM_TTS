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
import json
import logging
from io import BytesIO

import pickle
from collections import OrderedDict
import numpy as np
from pydub import AudioSegment
import traceback


def pad_to_multiple_of_val(arr, val=4):
    current_length = len(arr)
    remainder = current_length % val
    if remainder != 0:
        padding_length = val - remainder
        arr = np.pad(arr, [[0, padding_length], [0, 0] * (arr.ndim - 1)])
        # arr += [0] * padding_length  # 使用0进行填充，你可以根据需要更改填充值
    return arr


class FeatureLoader:
    def __init__(self, feature_keys,
                 max_tar_stream=50,
                 target_sr=22050) -> None:
        """
        code:
        - self.n_special_id
        - self.n_phone_id
        - self.ct.n_code
        - self.n_max_spkr
        """
        self.feature_keys = feature_keys
        self.tar_mgr = TarManager(max_tar_stream)
        self.target_sr=target_sr
        self.resample_buffer = dict()



    def get_item_key(self, k):
        if k in ['text', 'text_id', 'phone', 'audio_path', 'n_spkr']:
            src_k = 'info'
        elif k in ['wav_byte']:
            src_k = 'wav'
        else:
            src_k = k
        return src_k

    def __call__(self, item):
        loaded_sample = {}
        if 'uttid' in item:
            loaded_sample['uttid'] = item['uttid']
        for k in self.feature_keys:
            item_k = self.get_item_key(k)
            if item_k == 'emotion': # 无emotion标签情况
                if item_k not in item:
                    loaded_sample[k] = -1
                    continue
            try:
                feat_bytes = item[item_k]
                loaded_sample[k] = getattr(self, k)(feat_bytes)
            except:
                print(item[item_k])
                print(item)
            if k == 'text_id':  # 把text_id当成phone_id
                loaded_sample['phone_id'] = loaded_sample.pop(k)
            if k == 'prompt_speech': # path也要加入
                loaded_sample['prompt_speech_path'] = feat_bytes
        return loaded_sample

    # ############################################################

    def load_npy(self, stream):
        if isinstance(stream, bytes):
            stream = BytesIO(stream)
        data = np.load(stream)
        return data

    def load_json(self, stream):
        # with open(fp) as f:
        return json.loads(stream)

    def load_wav(self, wav_bytes, return_numpy=True):
        # wav_bytes = tar_reader.get(filename)
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format='wav')
        # write to disk
        # wav.export('test.wav', format='wav')

        if not return_numpy:
            return audio

        # 将AudioSegment转换为NumPy数组
        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        # 确定位深度
        bit_depth = audio.sample_width * 8  # sample_width是每个样本的字节数
        # 将样本值归一化到-1到1的范围内
        normalized_samples = samples / (2**(bit_depth - 1))
        # write to disk
        # soundfile.write('test_numpy.wav', normalized_samples, sr)

        assert sr == 24000  # tmp
        return normalized_samples
    
    # rl
    def prompt_text(self, fp):
        if not isinstance(fp, str) or '.json' in fp:
            return self.load_json(fp)['text']
        else:
            return fp
    
    def syn_text(self, fp):
        return fp
    
    # 返回地址
    def prompt_speech(self, fp):
        return fp
    
    def prompt_speech_token(self, fp):
        d = self.load_npy(fp)
        return d
    
    def prompt_speech_feat(self, fp):
        d = self.load_npy(fp)
        return d
    
    def embedding(self, fp):
        d = self.load_npy(fp)
        return d
    
    def emotion(self, fp):
        return fp