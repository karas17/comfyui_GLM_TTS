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
from hyperpyyaml import load_hyperpyyaml
import torch
from transformers import WhisperFeatureExtractor
import glob
import os
import safetensors
from utils.whisper_models.configuration_whisper import WhisperVQConfig
from utils.whisper_models.modeling_whisper import WhisperVQEncoder

def load_flow_model(flow_ckpt_path, config_path, device):
    with open(config_path, 'r') as f:
        scratch_configs = load_hyperpyyaml(f)
        flow = scratch_configs['flow']

    tmp = torch.load(flow_ckpt_path, map_location=device)
    if type(tmp) == dict:
        flow.load_state_dict(tmp["model"])

    else:
        flow.load_state_dict(tmp)

    flow.to(device)
    flow.eval()
    return flow

def load_quantize_encoder(model_path):
    print(f'[load_quantize_encoder] start. {model_path=}')
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model

def load_speech_tokenizer(model_path):
    model = load_quantize_encoder(model_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    return model, feature_extractor