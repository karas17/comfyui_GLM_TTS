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
import glob
import pdb
import io
import time
import torch
import torch.distributed as dist
from grpo.loaders.dataloader.dynamic_batch import BucketizerPipe
from grpo.loaders.dataloader.feature_loader_rl import FeatureLoader
from grpo.loaders.dataloader.loader4rl import Loader
from torch.utils.data import DataLoader
from functools import partial
import whisper
import os
from transformers import AutoTokenizer
import torchaudio
import sentencepiece as spm

global_need_mel_feature = False

def resample(audio_data, resample_rate=22050):
    sample = {}
    sample['speech'], sample['sample_rate'] = torchaudio.load(audio_data)
    sample_rate = sample['sample_rate']
    waveform = sample['speech']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['speech'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    max_val = sample['speech'].abs().max()
    if max_val > 1:
        sample['speech'] /= max_val
    return sample['speech']


def collate_fn_wo_frontend(item_list, codec_token_name, tknr_fn, text_frontend=None):
    t1 = time.time()
    """
    item_list:
    [ { 'text': ... }, {}, ... ]
    """
    assert isinstance(item_list, list)
    item_list = [item for item in item_list if item is not None]

    for item in item_list:
        # ['prompt_text', 'prompt_speech', 'prompt_speech_token', 'prompt_speech_feat', 'embedding', 'syn_text']
        text = text_frontend.text_normalize(item["prompt_text"])
        text_token = tknr_fn(text)
        item["prompt_text_token"] = text_token
        
        text = text_frontend.text_normalize(item["syn_text"])
        text_token = tknr_fn(text)
        item["syn_text_token"] = text_token
        
        # item["codec"] = item[codec_token_name].squeeze()

    # 过滤掉没有 'codec' 键的 item
    item_list = [item for item in item_list if 'prompt_speech_token' in item]
    # ========================================================= 进行pad
    # 1.speech token
    prompt_speech_token_list = [torch.tensor(obj["prompt_speech_token"], dtype=torch.long) for obj in item_list]
    # batch_prompt_speech_token = torch.nn.utils.rnn.pad_sequence(prompt_speech_token_list, batch_first=True, padding_value=0)
    prompt_speech_token_lengths = torch.LongTensor([len(item) for item in prompt_speech_token_list])

    # prompt_speech = [torch.tensor(obj["prompt_speech"], dtype=torch.float32) for obj in item_list]
    prompt_speech = [obj["prompt_speech"] for obj in item_list]
    # [b,d]
    prompt_text_list = [obj["prompt_text"] for obj in item_list]
    syn_text_list = [obj["syn_text"] for obj in item_list]
    prompt_text_token_list = [torch.tensor(obj["prompt_text_token"], dtype=torch.long) for obj in item_list]
    # batch_prompt_text_token = torch.nn.utils.rnn.pad_sequence(prompt_text_token_list, batch_first=True, padding_value=0)
    prompt_text_token_lengths = torch.LongTensor([len(item) for item in prompt_text_token_list])
    
    syn_text_token_list = [torch.tensor(obj["syn_text_token"], dtype=torch.long) for obj in item_list]
    # batch_syn_text_token = torch.nn.utils.rnn.pad_sequence(syn_text_token_list, batch_first=True, padding_value=0)
    syn_text_token_lengths = torch.LongTensor([len(item) for item in syn_text_token_list])
    
    mel_list = [torch.tensor(obj["prompt_speech_feat"].squeeze()[None, :], dtype=torch.float32) for obj in item_list]
    # batch_mel = torch.nn.utils.rnn.pad_sequence(mel_list, batch_first=True, padding_value=0)
    mel_lengths = torch.LongTensor([len(item) for item in mel_list])
    
    embedding_list = [torch.tensor(obj["embedding"].squeeze(), dtype=torch.float32) for obj in item_list]
    # batch_embedding = torch.Tensor(embedding_list)
    
    batch_uttid = [obj["uttid"] for obj in item_list]
    
    emotion_list = [torch.tensor(obj["emotion"], dtype=torch.long) for obj in item_list]
    output = {
        "uttid": batch_uttid,
        "prompt_speech_token": prompt_speech_token_list,
        # "prompt_speech_token_len": prompt_speech_token_lengths,
        "prompt_text_token": prompt_text_token_list,
        # "prompt_text_token_len": prompt_text_token_lengths,
        "speech_feat": mel_list,
        # "speech_feat_len": mel_lengths,
        "text": syn_text_list,
        "embedding": embedding_list,
        "syn_text_token": syn_text_token_list,
        "prompt_speech": prompt_speech,
        "emotion": emotion_list,
        # "syn_text_token_len": syn_text_token_lengths,
    }

    return output

def collate_fn_sft(item_list, codec_token_name, tknr_fn, text_frontend=None, embedding=torch.zeros(1, 192)):
    t1 = time.time()
    """
    item_list:
    [ { 'text': ... }, {}, ... ]
    """
    assert isinstance(item_list, list)
    item_list = [item for item in item_list if item is not None]

    for item in item_list:
        # ['prompt_text', 'prompt_speech', 'syn_text']
        text = ''
        item["prompt_text_token"] = torch.zeros(0, dtype=torch.int32)
        
        text = text_frontend.text_normalize(item["syn_text"])
        text_token = tknr_fn(text)
        item["syn_text_token"] = text_token
        
        # item["codec"] = item[codec_token_name].squeeze()

    
    # ========================================================= 进行pad
    # 1.speech token
    prompt_speech_token_list = [torch.zeros(0, dtype=torch.int32) for obj in item_list]
    # batch_prompt_speech_token = torch.nn.utils.rnn.pad_sequence(prompt_speech_token_list, batch_first=True, padding_value=0)
    prompt_speech_token_lengths = torch.LongTensor([len(item) for item in prompt_speech_token_list])

    # prompt_speech = [torch.tensor(obj["prompt_speech"], dtype=torch.float32) for obj in item_list]
    # prompt_speech_list = [torch.tensor(obj['prompt_speech']) for obj in item_list]
    # [b,d]
    syn_text_list = [obj["syn_text"] for obj in item_list]
    if "phone_id" in item_list[0]:
        assert "prompt_text_token" not in item_list[0]
        prompt_text_token_list = [torch.tensor(obj["phone_id"], dtype=torch.long) for obj in item_list]
    else:
        prompt_text_token_list = [obj["prompt_text_token"] for obj in item_list]
    # batch_prompt_text_token = torch.nn.utils.rnn.pad_sequence(prompt_text_token_list, batch_first=True, padding_value=0)
    prompt_text_token_lengths = torch.LongTensor([len(item) for item in prompt_text_token_list])
    
    if "phone_id" in item_list[0]:
        assert "syn_text_token" not in item_list[0]
        syn_text_token_list = [torch.tensor(obj["phone_id"], dtype=torch.long) for obj in item_list]
    else:
        syn_text_token_list = [torch.tensor(obj["syn_text_token"], dtype=torch.long) for obj in item_list]
    # batch_syn_text_token = torch.nn.utils.rnn.pad_sequence(syn_text_token_list, batch_first=True, padding_value=0)
    syn_text_token_lengths = torch.LongTensor([len(item) for item in syn_text_token_list])
    
    mel_list = [torch.zeros(1, 0, 80) for obj in item_list]
    # batch_mel = torch.nn.utils.rnn.pad_sequence(mel_list, batch_first=True, padding_value=0)
    mel_lengths = torch.LongTensor([len(item) for item in mel_list])
    
    embedding_list = [embedding for obj in item_list]
    # batch_embedding = torch.Tensor(embedding_list)
    
    batch_uttid = [obj["uttid"] for obj in item_list]
    
    emotion_list = [torch.tensor(obj["emotion"], dtype=torch.long) for obj in item_list]
    prompt_speech = [obj["prompt_speech"] for obj in item_list]
    # prompt_speech = ['None' for obj in item_list]
    output = {
        "uttid": batch_uttid,
        # "prompt_speech": prompt_speech,
        "prompt_speech_token": prompt_speech_token_list,
        # "prompt_speech_token_len": prompt_speech_token_lengths,
        "prompt_text_token": prompt_text_token_list,
        # "prompt_text_token_len": prompt_text_token_lengths,
        "speech_feat": mel_list,
        # "speech_feat_len": mel_lengths,
        "text": syn_text_list,
        "embedding": embedding_list,
        "syn_text_token": syn_text_token_list,
        "prompt_speech": prompt_speech,
        "emotion": emotion_list,
        # "syn_text_token_len": syn_text_token_lengths,
    }

    return output

def collate_fn_from_frontend(item_list, codec_token_name, tknr_fn, frontend=None, text_frontend=None, sample_rate=None):
    t1 = time.time()
    """
    item_list:
    [ { 'text': ... }, {}, ... ]
    """
    assert isinstance(item_list, list)
    item_list = [item for item in item_list if item is not None]

    for item in item_list:
        # ['prompt_text', 'prompt_speech', 'syn_text']
        text = text_frontend.text_normalize(item["prompt_text"])
        text_token = tknr_fn(text)
        item["prompt_text_token"] = text_token
        
        text = text_frontend.text_normalize(item["syn_text"])
        text_token = tknr_fn(text)
        item["syn_text_token"] = text_token
        
        # item["codec"] = item[codec_token_name].squeeze()

    
    # ========================================================= 进行pad
    # 1.speech token
    prompt_speech_token_list = [frontend._extract_speech_token_cpu([obj["prompt_speech_path"]]) for obj in item_list]
    # batch_prompt_speech_token = torch.nn.utils.rnn.pad_sequence(prompt_speech_token_list, batch_first=True, padding_value=0)
    prompt_speech_token_lengths = torch.LongTensor([len(item) for item in prompt_speech_token_list])

    prompt_speech_list = [torch.tensor(obj['prompt_speech']) for obj in item_list]
    # [b,d]
    prompt_text_list = [obj["prompt_text"] for obj in item_list]
    syn_text_list = [obj["syn_text"] for obj in item_list]
    if "phone_id" in item_list[0]:
        assert "prompt_text_token" not in item_list[0]
        prompt_text_token_list = [torch.tensor(obj["phone_id"], dtype=torch.long) for obj in item_list]
    else:
        prompt_text_token_list = [torch.tensor(obj["prompt_text_token"], dtype=torch.long) for obj in item_list]
    # batch_prompt_text_token = torch.nn.utils.rnn.pad_sequence(prompt_text_token_list, batch_first=True, padding_value=0)
    prompt_text_token_lengths = torch.LongTensor([len(item) for item in prompt_text_token_list])
    
    if "phone_id" in item_list[0]:
        assert "syn_text_token" not in item_list[0]
        syn_text_token_list = [torch.tensor(obj["phone_id"], dtype=torch.long) for obj in item_list]
    else:
        syn_text_token_list = [torch.tensor(obj["syn_text_token"], dtype=torch.long) for obj in item_list]
    # batch_syn_text_token = torch.nn.utils.rnn.pad_sequence(syn_text_token_list, batch_first=True, padding_value=0)
    syn_text_token_lengths = torch.LongTensor([len(item) for item in syn_text_token_list])
    
    mel_list = [frontend._extract_speech_feat_cpu(obj["prompt_speech_path"], sample_rate=sample_rate) for obj in item_list]
    # batch_mel = torch.nn.utils.rnn.pad_sequence(mel_list, batch_first=True, padding_value=0)
    mel_lengths = torch.LongTensor([len(item) for item in mel_list])
    
    embedding_list = [frontend._extract_spk_embedding_cpu(obj["prompt_speech_path"]) for obj in item_list]
    # batch_embedding = torch.Tensor(embedding_list)
    
    batch_uttid = [obj["uttid"] for obj in item_list]
    output = {
        "uttid": batch_uttid,
        "prompt_speech": prompt_speech_list,
        "prompt_speech_token": prompt_speech_token_list,
        # "prompt_speech_token_len": prompt_speech_token_lengths,
        "prompt_text_token": prompt_text_token_list,
        # "prompt_text_token_len": prompt_text_token_lengths,
        "speech_feat": mel_list,
        # "speech_feat_len": mel_lengths,
        "text": syn_text_list,
        "embedding": embedding_list,
        "syn_text_token": syn_text_token_list,
        # "syn_text_token_len": syn_text_token_lengths,
    }

    return output


def get_global_worker_info(worker_info=None):
    # 要让所有训练进程的所有worker都读不同的数据
    # 需要得到global_rank和world_size

    if torch.distributed.is_initialized():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    # worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print('Dataset executed without DataLoader.')
        local_worker_id = 0
        local_num_worker = 1
    else:
        local_worker_id = worker_info.id
        local_num_worker = worker_info.num_workers

    global_num_worker = local_num_worker * world_size
    global_worker_id = global_rank * local_num_worker + local_worker_id

    return global_worker_id, global_num_worker


def worker_init_fn(__worker_id):
    # 在创建dataset时执行, 功能是根据全局的worker_id (0~node*gpu*worker-1) 划分数据.
    # 背景: dataloader会先创建多个worker进程, 然后在worker进程里生成dataset实例.
    # 注意这个dataset实例不是从主进程里拷贝过来的, 而是重新创建的. 具体创建方式没太搞懂, 只记得和进程初始化方式有关.
    # 进程初始化有两种方式: fork和spawn.
    # 因此, 这部分会在创建dataset时执行.
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # 获取dataset副本
    if isinstance(dataset, Loader):
        real_dataset: Loader = dataset
    elif isinstance(dataset, BucketizerPipe):
        real_dataset: Loader = dataset.datapipe.datapipe
    else:
        raise

    # split all filelists
    global_worker_id, global_num_worker = get_global_worker_info(worker_info)
    # per_worker = int(np.ceil(
    #     len(real_dataset.meta_gen.all_json_files) / float(global_num_worker)
    # ))
    per_worker = int(
        len(real_dataset.meta_gen.all_json_files) / float(global_num_worker)
    )
    # print(f"all_json_files:{len(real_dataset.meta_gen.all_json_files)},global_num_worker:{global_num_worker},per_worker:{per_worker}")

    worker_id = global_worker_id
    # dataset.meta_gen.json_files = dataset.meta_gen.all_json_files[worker_id * per_worker:  (worker_id + 1) * per_worker]
    real_dataset.meta_gen.worker_init(worker_id * per_worker, (worker_id + 1) * per_worker, worker_id)


def get_train_loader(patterns, worker=1, max_token_in_batch=13000,
                     token_name="cosy_token", use_bucket=True, batch_size=16,
                     prefetch_factor=4, use_phone_id=False, tknr_fn=None, 
                     frontend=None, text_frontend=None, sample_rate=None, 
                     mode="PRETRAIN", sft_embedding=None, use_emo_tag=False, use_prompt=True):


    if mode == "LORA" or mode == "SFT" or not use_prompt:
        features = ['prompt_speech', 'syn_text']
    else:
        if frontend is None:
            features = ['prompt_text', 'prompt_speech', 'prompt_speech_token', 'prompt_speech_feat', 'embedding', 'syn_text']
        else:
            features = ['prompt_text', 'prompt_speech', 'syn_text']
    if use_emo_tag:
        features.append('emotion') 
    FL = FeatureLoader(features, target_sr=sample_rate)

    all_jsons = []
    for pat in patterns:
        # print(pat, len(pat))
        file_list = glob.glob(pat)
        assert len(file_list) > 0, "error: pattern no file:" + pat
        all_jsons += file_list

    print("all jsons:", len(all_jsons))

    pipe = Loader(
        patterns,
        FL,
    )

    if mode == "LORA" or mode == "SFT" or not use_prompt:
        collate_fn = partial(collate_fn_sft, codec_token_name=token_name, tknr_fn=tknr_fn, text_frontend=text_frontend, embedding=sft_embedding)
    elif frontend is None:
        collate_fn = partial(collate_fn_wo_frontend, codec_token_name=token_name, tknr_fn=tknr_fn, text_frontend=text_frontend)
    else:
        collate_fn = partial(collate_fn_from_frontend, codec_token_name=token_name, tknr_fn=tknr_fn, frontend=frontend, text_frontend=text_frontend, sample_rate=sample_rate)
    if use_bucket:
        def len_fn(data):
            # pdb.set_trace()
            # return len(data[token_name])
            return len(data['syn_text'])

        ds = BucketizerPipe(pipe, len_fn, batch_size=max_token_in_batch, bucket_size=200)
        batch_size = None
    else:
        ds = pipe

    dl = DataLoader(ds,
                    batch_size=batch_size,
                    num_workers=worker,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    prefetch_factor=None if worker == 0 else prefetch_factor,
                    worker_init_fn=worker_init_fn,
                    drop_last=True
                    )

    return dl

