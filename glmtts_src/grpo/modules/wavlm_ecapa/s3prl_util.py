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
import torch
import os


def load_feature_extract_for_wavlm_large():
    feature_extract = torch.hub.load('modules/s3prl',
                                     "wavlm_large",
                                     source='local',
                                     path="modules/s3prl/f2d5200177fd6a33b278b7b76b454f25cd8ee866d55c122e69fccf6c7467d37d.wavlm_large.pt",
                                     skip_validation=True,
                                     force_reload=True)
    return feature_extract
