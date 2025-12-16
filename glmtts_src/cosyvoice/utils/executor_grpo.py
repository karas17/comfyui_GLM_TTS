# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
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

import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist
import tqdm
from cosyvoice.utils.train_utils_grpo import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, \
    batch_backward, save_model, cosyvoice_join
from grpo.grpo_utils import normalize_rewards_per_group, rollout, normalize_rewards_per_group_token_level, normalize_rewards_per_group_norm_first
import numpy as np
import math

class Executor:

    def __init__(self, mode, step=0, epoch=0):
        self.step = step
        self.epoch = epoch
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))
        self.mode = mode

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict,
                       group_join, ref_model, reward_func):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        
        # GRPO settings
        NUM_ANSWERS_PER_QUESTION = info_dict['num_answers_per_question']
        
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in tqdm.tqdm(enumerate(train_data_loader), disable=self.rank != 0):
                try: 
                    info_dict["tag"] = self.mode
                    info_dict["step"] = self.step
                    info_dict["epoch"] = self.epoch
                    info_dict["batch_idx"] = batch_idx
                    
                    if info_dict.get('dynamic_T', False):
                        def temperature_scheduler(x):
                            if x <= info_dict['dynamic_T']['final_step']:
                                k = 0.005
                                return 1.0 + (info_dict['dynamic_T']['start'] - 1.0) * math.exp(-k * x)
                            else:
                                return 1.0
                        info_dict['generation_conf']['temperature'] = temperature_scheduler(self.step)
                    
                    def clip_eps_scheduler(step, max_step):
                            if step <= max_step:
                                return clip_info['start'] + (clip_info['end'] - clip_info['start']) * (step / max_step)
                            else:
                                return clip_info['end']
                            
                    if info_dict.get('dynamic_clip_eps_high', False):
                        clip_info = info_dict['dynamic_clip_eps_high']
                        info_dict['clip_eps_high'] = clip_eps_scheduler(self.step, clip_info['final_step'])
                    if info_dict.get('dynamic_clip_eps_low', False):
                        clip_info = info_dict['dynamic_clip_eps_low']
                        info_dict['clip_eps_low'] = clip_eps_scheduler(self.step, clip_info['final_step'])
                    if cosyvoice_join(group_join, info_dict):
                        break

                    # Disable gradient synchronizations across DDP processes.
                    # Within this context, gradients will be accumulated on module
                    # variables, which will later be synchronized.
                    if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                        context = model.no_sync
                    # Used for single gpu training and DDP gradient synchronization
                    # processes.
                    else:
                        context = nullcontext

                    local_rank = int(os.environ.get('LOCAL_RANK', 0))
                    device = torch.device("cuda", local_rank)
                    
                    max_try = 3
                    for attempt in range(max_try):
                        episodes = rollout(
                            model=model,
                            batch=batch_dict,
                            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                            reward_function=reward_func,
                            device=device,
                            info_dict=info_dict
                        )                  
                        normalized_episodes, has_grad = normalize_rewards_per_group(episodes, info_dict['reward_weight'])
                        if has_grad:
                            break
                    # if attempt > 0:
                    #     print('batch resampled, attempt num=', attempt+1)
                    
                    local_max_len = max(
                        len(episode.prefix_token_ids) + len(episode.generated_token_ids)
                        for episode in episodes
                    )
                    # 全局同步最大长度
                    tensor = torch.tensor([local_max_len]).to(device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                    max_len = tensor.item()
                    
                    with context():
                        info_dict = batch_forward(model, ref_model, batch_dict, info_dict, normalized_episodes, device, batch_max_length=max_len)
                        info_dict = batch_backward(model, info_dict)
                        # print('backward done')
                    
                    info_dict = update_parameter_and_lr(model, optimizer, scheduler, info_dict)
                    log_per_step(writer, info_dict)

                    
                    # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                    if (info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and (
                            batch_idx + 1) % info_dict["accum_grad"] == 0):
                        dist.barrier()
                        self.cv(model, optimizer, scheduler, cv_data_loader, writer, info_dict, on_batch_end=False)
                        if (self.step + 2) > info_dict["final_step"]:
                            assert False, f"{self.mode} mode enough, steps: {self.step}"
                        model.train()
                    if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                        self.step += 1

                        if torch.distributed.get_rank() == 0 and info_dict["step"] % 10 == 0:
                            grad_norm = info_dict["grad_norm"]
                            if isinstance(grad_norm, torch.Tensor):
                                grad_norm = grad_norm.item()

                            my_info = {
                                "train/epoch": info_dict["epoch"],
                                "train/step": info_dict["step"],
                                "train/loss": info_dict["loss_dict"]["loss"].item(),
                                "train/lr": info_dict["lr"],
                                "train/grad_norm": grad_norm,
                                "train/resample_num": attempt + 1,
                            }
                            if "acc" in info_dict["loss_dict"]:
                                my_info["train/acc"] = info_dict["loss_dict"]["acc"].item()
                            if "text_loss" in info_dict["loss_dict"]:
                                my_info["train/text_loss"] = info_dict["loss_dict"]["text_loss"].item()
                            if "speech_loss" in info_dict["loss_dict"]:
                                my_info["train/speech_loss"] = info_dict["loss_dict"]["speech_loss"].item()
                            if "text_acc" in info_dict["loss_dict"]:
                                my_info["train/text_acc"] = info_dict["loss_dict"]["text_acc"].item()
                            if "speech_acc" in info_dict["loss_dict"]:
                                my_info["train/speech_acc"] = info_dict["loss_dict"]["speech_acc"].item()
                            if "loss" in info_dict["loss_dict"]:
                                my_info["train/loss"] = info_dict["loss_dict"]["loss"].item()
                            if "kl_loss" in info_dict["loss_dict"]:
                                my_info["train/kl_loss"] = info_dict["loss_dict"]["kl_loss"].item()
                            if "grpo_loss" in info_dict["loss_dict"]:
                                my_info["train/grpo"] = info_dict["loss_dict"]["grpo_loss"].item()
                            if "ratio_clip_low" in info_dict["loss_dict"]:
                                my_info["train/ratio_clip_low"] = info_dict["loss_dict"]["ratio_clip_low"].item()
                            if "ratio_clip_high" in info_dict["loss_dict"]:
                                my_info["train/ratio_clip_high"] = info_dict["loss_dict"]["ratio_clip_high"].item()
                            if 'sim_reward' in episodes[0].reward_info:
                                sim_reward = [
                                    episode.reward_info["sim_reward"] for episode in episodes
                                ]
                                sim_reward_mean = np.mean(sim_reward)
                                sim_reward_var = np.std(sim_reward)
                                my_info["train/sim_reward"] = sim_reward_mean
                                my_info["train/sim_reward_std"] = sim_reward_var
                            if 'cer_reward' in episodes[0].reward_info:
                                cer_reward = [
                                    episode.reward_info["cer_reward"] for episode in episodes
                                ]
                                cer_reward_mean = np.mean(cer_reward)
                                cer_reward_var = np.std(cer_reward)
                                my_info["train/cer_reward"] = cer_reward_mean
                                my_info["train/cer_reward_std"] = cer_reward_var
                            if 'laughter_reward' in episodes[0].reward_info:
                                laughter_reward = [
                                    episode.reward_info["laughter_reward"] for episode in episodes
                                ]
                                laughter_reward_mean = np.mean(laughter_reward)
                                laughter_reward_var = np.std(laughter_reward)
                                my_info["train/laughter_reward"] = laughter_reward_mean
                                my_info["train/laughter_reward_std"] = laughter_reward_var
                            if 'emo_reward' in episodes[0].reward_info:
                                emo_reward = [
                                    episode.reward_info["emo_reward"] for episode in episodes
                                ]
                                emo_reward_mean = np.mean(emo_reward)
                                emo_reward_var = np.std(emo_reward)
                                my_info["train/emo_reward"] = emo_reward_mean
                                my_info["train/emo_reward_std"] = emo_reward_var
                                # emo_reward = np.array(emo_reward)
                                emo_dict = {0:'angry', 6:'sad', 3:'happy', 7:'surprised', 2:'fear'}
                                # sample_num = 1
                                emolabel = batch_dict['emotion'][0].item()
                                if emolabel in emo_dict:
                                    my_info[f"train/emo_reward_{emo_dict[emolabel]}"] = emo_reward_mean
                                    my_info[f"train/emo_reward_{emo_dict[emolabel]}_std"] = emo_reward_var
                                
                                emo_neg_reward = [
                                    episode.reward_info["emo_neg_reward"] for episode in episodes
                                ]
                                emo_neg_reward = np.mean(emo_neg_reward)
                                my_info["train/emo_neg_reward"] = emo_neg_reward

                            print(my_info)
                except Exception as e:
                    print(f"Error in train_one_epoc: {e}")
                    print("Batch info: ", batch_dict)
                    raise e

        dist.barrier()
        self.cv(model, optimizer, scheduler, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, optimizer, scheduler, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch,
                                                                                                        self.step + 1)
        save_model(model, optimizer, scheduler, info_dict, model_name)
