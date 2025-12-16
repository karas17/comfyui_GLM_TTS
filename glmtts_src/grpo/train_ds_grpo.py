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
from __future__ import print_function
import os
import debugpy
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
import pdb
import wandb
import tqdm
import deepspeed
import datetime
from copy import deepcopy
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import ast
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import torch
import torch.distributed as dist
import yaml
import argparse
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.executor_grpo import Executor
from cosyvoice.utils.train_utils_grpo import (
    init_distributed,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)

import random
import numpy as np

from typing import Any, Dict, List, Optional

from glmtts_inference import load_models
from grpo.reward_func import reward_function_server
from glmtts_inference import get_special_token_ids
import torchaudio


SEED = 10086
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from functools import partial

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 对某些 CUDA 操作必要

def is_main_world():
    if args.multinode:
        return dist.get_rank() == 0
    return True


def print_model_parameters(model, name=""):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name, 'parameters: %.2f B' % (num_params / 10000 / 100 / 1000))


def get_core_model():
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        the_model = model.module
    else:
        the_model = model
    return the_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_train_setting(folder_path, args):
    config_file = args.config
    import yaml
    def generic_constructor(loader, tag_suffix, node):
        # 将自定义标签转换为包含标签信息的字典
        return {f"!{tag_suffix}": loader.construct_mapping(node)}
    # 注册通用构造函数处理所有自定义标签
    yaml.add_multi_constructor('', generic_constructor, Loader=yaml.SafeLoader)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    transformer_cfg = next(iter(config['llm'].values()))  # 获取第一个键对应的值
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'train_setting.json'), 'w') as f:
        dict_to_save = {
            "mode": args.mode,
            "name": args.name,
            "data-patterns": args.data_patterns,
            "text_tokenizer": args.text_tokenizer,
            "config_path": args.config,
            "checkpoint": args.checkpoint,
            "llm": transformer_cfg,
            "train_conf": config['train_conf']
        }
        if args.mode == "LORA":
            with open(transformer_cfg["lora_adapter_config"]) as f_lora:
                lora_adapter_config = json.load(f_lora)
            dict_to_save['lora_adapter_config'] = lora_adapter_config
        f.write(json.dumps(dict_to_save, ensure_ascii=False, indent=2) + '\n')
        f.flush()

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='deepspeed',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', default="llm", help='model which will be trained')
    parser.add_argument('--project', default="")
    parser.add_argument('--name', default="")
    parser.add_argument('--text_tokenizer', default="")
    parser.add_argument('--mode', default='PRETRAIN', choices=["PRETRAIN", "SFT", "LORA"])
    parser.add_argument('--save_root', default="./ckpt")
    parser.add_argument("--dryrun", default=True, type=ast.literal_eval)
    parser.add_argument("--use_phone", default=False, type=ast.literal_eval)
    parser.add_argument("--use_prompt", default=True, type=ast.literal_eval)

    parser.add_argument('--config', help='config file', default='ckpt/llm/config.json')
    parser.add_argument('--data-patterns', default='data/rl.yaml')
    # parser.add_argument('--train_data', required=True, help='train data file')
    # parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--checkpoint',default='',
                         help='checkpoint model')
    parser.add_argument('--model_dir', default="", required=False, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=1,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--max_token_in_batch',
                        default=100,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--resume_step',
                        default=0,
                        type=int,
                        )
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=True,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--accumulate_gradient', default=1, type=int)
    parser.add_argument('--multinode', default=False, type=ast.literal_eval)
    parser.add_argument('--use_amp', default=True, type=ast.literal_eval)
    parser.add_argument('--worker', default=1, type=int)
    parser.add_argument('--seed', default=10086, type=int)
    parser.add_argument('--resume', help='if continue training, if so set true, otherwise set false', default=False, type=ast.literal_eval)
    
    parser.add_argument("--flow", default="", type=str)
    parser.add_argument("--sample_method", default="ras", type=str)
    parser.add_argument("--lora_ckpt_path", default=None, type=str)
    parser.add_argument("--use_cache", default='True', choices=['True', ''])
    parser.add_argument("--stream", action="store_true")
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    args.model_dir = os.path.join(args.save_root, args.project, args.name)
    save_train_setting(args.model_dir, args)
    # =============================================================== wb日志、seed、多机初始化
    world_size, local_rank, rank = init_distributed(args)
    if not args.multinode:
        device = torch.device("cuda")
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device("cuda", local_rank)

    # =============================================================== model
    
    sample_rate = 24000
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(use_phoneme=False)

    del speech_tokenizer, frontend.campplus_session
    torch.cuda.empty_cache()  # 清空未用显存
    import gc
    gc.collect()        # 清理无主对象
    torch.cuda.empty_cache()  # 再保守一次清理
    

    reward_func = partial(reward_function_server, flow=flow, server_url="http://172.18.68.109:808")
    
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)

    # deepspeed
    configs['train_conf'].update(vars(args))
    configs = check_modify_and_save_config(args, configs)
    # Tensorboard summary
    writer = init_summarywriter(args)

    # model = configs[args.model]
    model = llm

    print_model_parameters(model)
    
    # ------------------ GRPO
    
    if args.mode == "LORA" or args.mode == "SFT":
        embedding = np.load(flow.infer_emb_path)
        embedding = torch.from_numpy(embedding).cpu()
        embedding = embedding.float()
    else:
        embedding = None
        
    ref_model = deepcopy(model)
    ref_model = ref_model.to(torch.device("cuda", local_rank))
    
    # ------------------ train
    model = model.cpu() # 在deepspeed.initialize之前放回cuda上
    model = wrap_cuda_model(args, model)
    current_step = 0
        
    if args.multinode:
    # 确保所有节点完成初始化
       dist.barrier()
       if dist.get_rank() == 0:
           print(f"World size: {dist.get_world_size()} initialized")
    model, optimizer, scheduler = init_optimizer_and_scheduler(args, configs, model)
    # import pdb; pdb.set_trace()
    if args.checkpoint is not None and args.mode != "LORA":
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        if args.train_engine.lower() == "deepspeed":
            print("DEEPSPEED: load_checkpoint...")
            if args.resume:
                _, client_state = model.load_checkpoint(args.checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True)
                current_step = client_state.get('step', 0)
            else:
                _, client_state = model.load_checkpoint(args.checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False)
                current_step = 0
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            current_step = checkpoint['step']

    if is_main_world():
        total = 0
        trainable = 0
        for n,p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"参数统计: 总参数 {total/1e6:.2f}M, 可训练参数 {trainable/1e6:.2f}M")
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    # =============================================================== loader
    with open(args.data_patterns, 'r') as f:
        patterns = yaml.safe_load(f)
        patterns = list(patterns.values())[0]
    
    from grpo.loaders import loader_lm_rl

    
    train_data_loader = loader_lm_rl.get_train_loader(
        patterns,
        args.worker,
        # max_token_in_batch=args.max_token_in_batch, 
        batch_size=1, use_bucket=False,
        use_phone_id=False, tknr_fn=frontend.tokenize_fn,
        frontend=None, text_frontend=text_frontend, sample_rate=sample_rate,
        mode=args.mode, sft_embedding=embedding, use_emo_tag=True, use_prompt=args.use_prompt)
    # =============================================================== training...
    if is_main_world():
        print("patterns:", len(patterns))
        print("Start training loop...")
    # train(init_steps)
    # Get executor
    info_dict = deepcopy(configs['train_conf'])
    if args.mode == "SFT" or args.mode == "LORA":
        spk = args.flow.split('flow')[0]
        info_dict['spk'] = spk
    executor = Executor(mode = args.mode, step=current_step, epoch=0)

    for epoch in range(info_dict['max_epoch']):
        executor.epoch = epoch
        # train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, None, writer, info_dict,
                                group_join, ref_model=ref_model, reward_func=reward_func)
        dist.destroy_process_group(group_join)

