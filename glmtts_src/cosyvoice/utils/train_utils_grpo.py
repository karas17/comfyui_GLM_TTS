# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pdb
from contextlib import nullcontext
import logging
import os
import math
import torch
import json
import re
import datetime
import yaml
import numpy as np
from peft import get_peft_model_state_dict
import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR, CosineAnnealing
from torch.optim.lr_scheduler import CosineAnnealingLR

from grpo.grpo_utils import normalize_rewards_per_group, compute_entropy, compute_kl_loss
from cosyvoice.utils.common import IGNORE_ID

def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    assert local_rank >= 0
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        try:
            torch.cuda.set_device(local_rank)
        except Exception as e:
            print("wrong device, local_rank:", local_rank)
            raise e
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank



def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        # print("skip modify dtype")
        configs['train_conf']["dtype"] = 'fp32'
    else:
        print("use deepspeed: ", args.deepspeed_config)
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(
            configs['train_conf']['save_per_step'] * configs['train_conf']['accum_grad'] / ds_configs[
                "gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model):
    params = model.parameters()
    if configs['train_conf']['optim'] == 'adam':
        optimizer = optim.Adam(params, **configs['train_conf']['optim_conf'])
    elif configs['train_conf']['optim'] == 'adamw':
        optimizer = optim.AdamW(params, **configs['train_conf']['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['train_conf'])

    if configs['train_conf']['scheduler'] == 'warmuplr':
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'constantlr':
        scheduler_type = ConstantLR
        scheduler = ConstantLR(optimizer)
    elif configs['train_conf']['scheduler'] == 'cosine':
        print("configs['train_conf']['scheduler_conf'].get('min_lr', 0):  ", configs['train_conf']['scheduler_conf'].get('min_lr', 0))
        print("configs['train_conf']['scheduler_conf']['max_steps']:  ", configs['train_conf']['scheduler_conf']['max_steps'])
        scheduler_type = CosineAnnealing
        scheduler = CosineAnnealing(
            optimizer, 
            max_steps=configs['train_conf']['scheduler_conf']['max_steps'], 
            min_lr=configs['train_conf']['scheduler_conf'].get('min_lr', 0),
        )
        

    else:
        raise ValueError("unknown scheduler: " + configs['train_conf'])

    # use deepspeed optimizer for speedup
    if args is not None and args.train_engine == "deepspeed":
        def scheduler(opt):
            if configs['train_conf']['scheduler'] == "constantlr":
                return scheduler_type(opt)
            else:
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])

        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=model.parameters())

    return model, optimizer, scheduler


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer

def save_model(model, optimizer, scheduler, info_dict, model_name = 'init'):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]

    # ckpt_delete_util.delete_ckpt(model_dir)
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            # ckpt_delete_util.delete_ckpt(model_dir, keep_latest=15, keep_every=50000)
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': info_dict['step'],
            }, save_model_path)

    else:
        # ckpt_delete_util.delete_folder(model_dir, keep_latest=15, keep_every=50000)
        with torch.no_grad():
            if info_dict["tag"] == "LORA":
                lora_state = get_peft_model_state_dict(model.llama)
                torch.save(lora_state, os.path.join(model_dir, f"{model_name}_lora.pt"))
            else:
                model.save_checkpoint(save_dir=model_dir,
                                    tag=model_name,
                                    client_state=info_dict)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))

def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    assert local_rank >= 0
    rank = int(os.environ.get('RANK', 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)
            return False
        except RuntimeError as e:
            logging.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True
    else:
        return False


def batch_forward(model, ref_model, batch, info_dict, episodes, device,
                  micro_batch_size=100,
                  batch_max_length = 750):
    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32
    if info_dict['train_engine'] == 'torch_ddp':
        autocast = nullcontext()
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    """Update the policy using the GRPO algorithm."""
    # episodes = normalize_rewards_per_group(episodes, info_dict['reward_weight'])
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    
    # entropy = torch.tensor(0.0, device=device)
    # grpo_loss = torch.tensor(0.0, device=device)
    # kl_loss = torch.tensor(0.0, device=device)
    info_dict['loss_dict'] = {}
    
    batch_episodes = episodes
    batch_lengths = [
        len(episode.prefix_token_ids) + len(episode.generated_token_ids)
        for episode in batch_episodes
    ]
    # batch_max_length = max(batch_lengths)
    batch_token_ids = [
        episode.prefix_token_ids
        + episode.generated_token_ids
        + [model.pad] * (batch_max_length - batch_lengths[i])
        for i, episode in enumerate(batch_episodes)
    ]
    batch_target_token_ids = [
        episode.prefix_token_ids
        + episode.generated_token_ids
        + [IGNORE_ID] * (batch_max_length - batch_lengths[i])
        for i, episode in enumerate(batch_episodes)
    ]
    batch_masks = [
        [0] * len(episode.prefix_token_ids)
        + [1] * len(episode.generated_token_ids)
        + [0] * (batch_max_length - batch_lengths[i])
        for i, episode in enumerate(batch_episodes)
    ]
    # batch_advantages = [episode.reward for episode in batch_episodes]
    batch_advantages = [
        [0] * len(episode.prefix_token_ids)
        + episode.reward
        + [0] * (batch_max_length - batch_lengths[i])
        for i, episode in enumerate(batch_episodes)
    ]
    
    all_token_ids = np.array(batch_token_ids).flatten()
    illegal = (all_token_ids != IGNORE_ID) & (all_token_ids != model.pad) & ((all_token_ids < 0) | (all_token_ids >= model.llama.vocab_size))
    if illegal.any():
        raise ValueError(f"Found {illegal.sum()} illegal token ids out of vocab range!")
    
    batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
    batch_target_token_ids = torch.tensor(batch_target_token_ids, device=device, dtype=torch.long)
    batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
    batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)
    
    # reward_keys = list(batch_episodes[0].reward_info.keys())
    # batch_id_advantanges = {k: torch.tensor([episode.reward_info[k] for episode in batch_episodes], device=device, dtype=torch.float32) for k in reward_keys}
    
    with autocast:
        input_token_ids = batch_token_ids[:, :-1]
        target_token_ids = batch_target_token_ids[:, 1:]
        target_masks = batch_masks[:, 1:]
        target_advantages = batch_advantages[:, 1:]
        logits = model.llama.forward(input_token_ids)["logits"].float()
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=IGNORE_ID,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)
        with torch.no_grad():
            ref_logits = ref_model.llama.forward(input_token_ids)["logits"].float()
            ref_log_probs = -torch.nn.functional.cross_entropy(
                ref_logits.reshape(-1, logits.size(-1)),
                target_token_ids.reshape(-1),
                ignore_index=IGNORE_ID,
                reduction="none",
            ).reshape(input_token_ids.shape[0], -1)
        
        # with torch.no_grad():
        #     token_entropy = compute_entropy(logits)
        #     entropy = (token_entropy * target_masks).sum() / num_target_tokens
        
        log_ratio = log_probs - ref_log_probs
        
        policy_loss_type = info_dict.get('policy_loss_type', 'ppo')
        if policy_loss_type == "ppo":
            ratio = log_ratio.exp()
        elif policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            ratio = (log_ratio * target_masks).sum(dim=-1) / target_masks.sum(dim=-1)
            ratio = ratio.exp().unsqueeze(-1) * target_masks
            
        
        surr1 = ratio * target_advantages
        
        clip_eps_low = info_dict.get('clip_eps_low', 0.2)
        clip_eps_high = info_dict.get('clip_eps_high', 0.2)
        surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * target_advantages


        micro_grpo_loss = -torch.min(surr1, surr2)

        micro_grpo_loss = (micro_grpo_loss * target_masks).sum() / num_target_tokens
        
        grpo_loss = micro_grpo_loss
        # clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        
        micro_kl_loss = compute_kl_loss(log_probs, ref_log_probs)
        micro_kl_loss = (micro_kl_loss * target_masks).sum() / num_target_tokens
        kl_loss = micro_kl_loss
        
        with torch.no_grad():
            clip_low = 1 - clip_eps_low
            clip_high = 1 + clip_eps_high
            ratio_clip_low = ((ratio < clip_low) * target_masks).sum() / num_target_tokens
            ratio_clip_high = ((ratio > clip_high) * target_masks).sum() / num_target_tokens
            
         
    info_dict['loss_dict']["loss"] = grpo_loss + info_dict['kl_weight'] * kl_loss
    # info_dict['loss_dict']["loss"] = (ratio * target_masks).sum() / num_target_tokens # only for debug
    info_dict['loss_dict']["grpo_loss"] = grpo_loss
    info_dict['loss_dict']["kl_loss"] = kl_loss
    info_dict['loss_dict']["ratio_clip_low"] = ratio_clip_low
    info_dict['loss_dict']["ratio_clip_high"] = ratio_clip_high
    # info_dict['entropy'] = entropy
    
    return info_dict


def batch_backward(model, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
        if torch.isfinite(grad_norm):
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
                (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict[
                    'accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # PRETRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "PRETRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step + 1, lr, rank, ' '.join(['{}_{}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)