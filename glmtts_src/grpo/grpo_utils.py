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
import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from grpo.data_types import Episode, MiniBatch
from llm.glmtts import GLMTTS
from cosyvoice.utils.common import IGNORE_ID
import torchaudio
import torch.nn.functional as F
import os
@torch.no_grad()
def batch_inference(
    model: GLMTTS, 
    batch, 
    num_answer_per_question: int,
    device: torch.device,
    max_gen_len: int = 1000,
    left_pad_id: int = 59246,
    sampling: int = 25,
    topp: float = 0.9,
    temperature: float = 0.95,
    sample_method: str = "ras",
    spk: str = None
    ):
    """
    Enhanced batch inference that generates n samples at once.
    Uses batch inference with dynamic left-padding and attention masking for all cases.
    """
    bsz = len(batch['uttid'])
    generated_token_ids = []
    prefix_token_ids = []
    group_ids = []
    
    # Prepare all input sequences for batch processing
    all_input_tokens = []
    all_prefix_tokens = []
    all_uttids = []  
    
    for k in range(bsz):
        for t in range(num_answer_per_question):
            uttid = batch['uttid'][k]
            prompt_text_token = batch['prompt_text_token'][k].to(device)
            tts_text_token = batch['syn_text_token'][k].to(device)
            prompt_speech_token = batch['prompt_speech_token'][k].to(device)
            
            # Create input sequence
            if model.mode == "SFT":
                if model.spk_prompt_dict is not None:
                    spk_prompt = model.spk_prompt_dict[spk]
                    # print(f"=========== llm_llama_glm.py, spk: {spk}, spk_prompt: {spk_prompt} ===========")
                    input_tokens = torch.cat([
                        torch.tensor(spk_prompt).to(device), 
                        prompt_text_token, 
                        tts_text_token, 
                        torch.tensor([model.boa]).to(device),
                        prompt_speech_token + model.ats]).to(torch.long)
                else:
                    raise ValueError(f"Invalid mode: {model.mode}")
            elif model.mode == "PRETRAIN" or model.mode == "LORA":            
                input_tokens = torch.cat([
                    prompt_text_token,
                    tts_text_token,
                    torch.tensor([model.boa]).to(device),
                    prompt_speech_token + model.ats
                ])
            
            prefix_token = input_tokens

            all_input_tokens.append(input_tokens)
            all_prefix_tokens.append(prefix_token)
            all_uttids.append(uttid)
    
    # Batch process with dynamic left-padding (moved outside the loops)
    batch_size = len(all_input_tokens)
    
    # Find max length in the batch
    max_len = max(len(tokens) for tokens in all_input_tokens)
    # import pdb; pdb.set_trace()
    # Create padded input tensor (pad to max_len only, no fixed padding)
    padded_input_tokens = torch.full(
        (batch_size, max_len), 
        left_pad_id, 
        dtype=torch.long, 
        device=device
    )
    
    # Fill in the actual tokens (left-padded to max_len)
    for i, tokens in enumerate(all_input_tokens):
        pad_length = max_len - len(tokens)
        if pad_length > 0:
            # Left pad with left_pad_id, then fill actual tokens
            padded_input_tokens[i, pad_length:] = tokens
        else:
            # No padding needed
            padded_input_tokens[i, :] = tokens
    # import pdb; pdb.set_trace()
    # Create attention mask: 0 for padding, 1 for real tokens
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, tokens in enumerate(all_input_tokens):
        pad_length = max_len - len(tokens)
        attention_mask[i, pad_length:] = 1  # Real tokens get attention
    
    # Get embeddings
    # lm_input = self.llama_embedding(llm_input_token_pad)
    inputs_embeds = model.llama_embedding(padded_input_tokens)
    
    # Batch generation with early stopping per sample
    generated_sequences = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    past_key_values = None
    current_attention_mask = attention_mask.clone()
    
    for step in range(max_gen_len):
        if all(finished):
            break
            
        # Forward pass
        model_input = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": current_attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": True,
            "past_key_values": past_key_values
        }
        
        outputs = model.llama(**model_input)
        past_key_values = outputs['past_key_values']
        logits = outputs['logits'][:, -1]  # [batch_size, vocab_size]
        
        # Sample next tokens for each sequence
        next_tokens = []
        for i in range(batch_size):
            if finished[i]:
                next_tokens.append(model.pad)  # Use pad token for finished sequences
                continue
                
            logp = logits[i].log_softmax(dim=-1)
            
            if sample_method == "ras":
                next_token = model.sampling_ids_ras(logp, generated_sequences[i], sampling).item()
                # next_token = model.sampling_ids_ras(logp, generated_sequences[i], sampling, topp, temperature).item()
            else:
                # Default to topk sampling
                next_token = model.sampling_ids(logp, sampling, 1).item()
            
            if next_token == model.eoa:
                finished[i] = True
                next_tokens.append(model.pad)  # Use pad token
            else:
                generated_sequences[i].append(next_token)
                next_tokens.append(next_token)
        
        # Prepare next iteration inputs
        next_token_tensor = torch.tensor(next_tokens, device=device).unsqueeze(1)
        next_token_embeds = model.llama_embedding(next_token_tensor)
        inputs_embeds = next_token_embeds
        
        # Extend attention mask
        new_attention = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
    
    # Convert generated sequences to tensors and collect results
    for i in range(batch_size):
        if generated_sequences[i]:
            generated_token_id = torch.tensor(generated_sequences[i], device=device)
        
        generated_token_ids.append(generated_token_id)
        prefix_token_ids.append(all_prefix_tokens[i])
        group_ids.append(all_uttids[i])
    
    return generated_token_ids, prefix_token_ids, group_ids        

   
    
    
@torch.no_grad()
def rollout(
    model: GLMTTS,
    batch,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    info_dict: dict
) -> List[Episode]:
    '''
        batch = {
        "uttid": batch_uttid,
        "prompt_speech_token": prompt_speech_token_list,
        "prompt_text_token": prompt_text_token_list,
        "speech_feat": mel_list,
        "text": syn_text_list,
        "embedding": embedding_list,
        "syn_text_token": syn_text_token_list,
    }
    '''
    bsz = len(batch['uttid']) * num_answer_per_question
    
    generation_conf = info_dict['generation_conf']
    generated_token_ids_list, prefix_token_ids_list, group_ids_list = batch_inference(model, batch, num_answer_per_question, device,
                                                                                      topp=generation_conf['topp'],
                                                                                      temperature=generation_conf['temperature'],
                                                                                      spk=info_dict.get('spk', None),
                                                                                      )
    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = generated_token_ids_list[idx]
            prefix_token_ids = prefix_token_ids_list[idx]
            group_ids = group_ids_list[idx]

            '''
            def reward_function(
                    response_token: List[int],
                    prompt_speech_token: torch.Tensor,
                    speech_feat: torch.FloatTensor,
                    embedding: torch.FloatTensor,
                    target_audio: torch.FloatTensor,
                ) -> Dict[str, Any]:
            '''
            save_name = f'{group_ids}_{j}'
            rewards = reward_function(
                save_name,
                response_token=(generated_token_ids - model.ats).tolist(), # 进入flow时要把token的偏移减掉
                prompt_speech_token=batch['prompt_speech_token'][i].unsqueeze(0),
                speech_feat=batch['speech_feat'][i],
                embedding=batch['embedding'][i].unsqueeze(0),
                target_audio=batch['prompt_speech'][i] if 'prompt_speech' in batch else None,
                ref_text=batch['text'][i],
                emotion=batch['emotion'][i],
            )

            episode = Episode(
                # prefix=batch.prefix[i],
                prefix_token_ids=prefix_token_ids.tolist(),
                # prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids.tolist(),
                group_token_ids=group_ids,
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group_norm_first(episodes: List[Episode], reward_weights: dict = None) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_token_ids].append(episode)
    output = []
    has_grad = False
    for group in groups.values():
        reward_keys = list(group[0].reward_info.keys())
        reward_weight_sum = sum(reward_weights[k] for k in reward_keys)
        group_rewards = [
            sum(
                reward_weights[k] / max(item.reward_info[k], 1e-12) for k in reward_keys if reward_weights[k] != 0
            ) / reward_weight_sum
            for item in group
        ]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        if std_reward > 1e-2:
            has_grad = True
        for idx, episode in enumerate(group):
            normalized_reward = (group_rewards[idx] - mean_reward) / (std_reward + 1e-4)
            if isinstance(normalized_reward, float):
                normalized_reward = [normalized_reward] * len(episode.generated_token_ids)   
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output, has_grad

def normalize_rewards_per_group(episodes: List[Episode], reward_weights: dict = None) -> List[Episode]:
    """
    对reward_info下各项分别normalize，再加和
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_token_ids].append(episode)

    output = []
    has_grad = True
    for group in groups.values():
        # 找所有reward_info下的key（假定每条都有完整reward_info）
        reward_keys = list(group[0].reward_info.keys())
        reward_keys = [x for x in reward_keys if x!='token_cer_reward'] # 处理不了token_level_cer
        # 收集所有分项reward
        reward_array = {k: np.array([ep.reward_info[k] for ep in group]) for k in reward_keys}
        normed_rewards = {}

        # 单项归一化
        for k in reward_keys:
            arr = reward_array[k]
            mean = arr.mean()
            std = arr.std()
            if std < 1e-2:
                # print('no grad samples, ', k)
                normed_rewards[k] = [0.0 for v in arr]
            else:
                normed_rewards[k] = [(v - mean)/std for v in arr]

        # 加和，再整体归一化（如有需要）
        if reward_weights is None:
            summed = [sum(normed_rewards[k][i] for k in reward_keys) for i in range(len(group))]
        else:
            summed = [sum(normed_rewards[k][i] * reward_weights[k] for k in reward_keys if k in reward_weights) for i in range(len(group))]
        mean_summed = np.mean(summed)
        std_summed = np.std(summed)

        for idx, episode in enumerate(group):
            if std_summed < 1e-2:
            # if False:
                summed_norm = 0.0
                has_grad = False
            else:
                summed_norm = (summed[idx] - mean_summed) / std_summed
            
            if isinstance(summed_norm, float):
                summed_norm = [summed_norm] * len(episode.generated_token_ids)
            # 更新归一化后的各项（如要保留）
            new_reward_info = episode.reward_info.copy()
            for k in reward_keys:
                new_reward_info[k] = normed_rewards[k][idx]
            # 新reward可以是 summed[idx] 或 summed_norm，视实验需求
            episode_new = dataclasses.replace(episode, reward=summed_norm, reward_info=new_reward_info)
            output.append(episode_new)
    return output, has_grad


def normalize_rewards_per_group_token_level(episodes: List[Episode], reward_weights: dict = None) -> List[Episode]:
    """
    对reward_info下各项分别normalize，再加和
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_token_ids].append(episode)

    output = []
    has_grad = False
    for group in groups.values():
        # 找所有reward_info下的key（假定每条都有完整reward_info）
        reward_keys = list(group[0].reward_info.keys())
        # 收集所有分项reward
        reward_array = {k: [ep.reward_info[k] for ep in group] for k in reward_keys}
        normed_rewards = {}

        # 单项归一化
        for k in reward_keys:
            arr = reward_array[k]
            if isinstance(arr[0], list):
                result = []
                for l in arr:
                    result.extend(l)
                result = np.array(result)
                mean = result.mean()
                std = result.std()
            else:
                arr = np.array(arr)
                mean = arr.mean()
                std = arr.std()
            if std < 1e-2:
                # print('no grad samples, ', k)
                # arr: 为了计算时兼容
                normed_rewards[k] = [np.array(0.0) for v in arr]
            else:
                if ('emo' in k or 'cer' in k) and reward_weights[k] != 0:
                    has_grad = True
                    # print(k, arr)
                if k == 'token_cer_reward': # 不norm
                    normed_rewards[k] = [np.array(v) for v in arr]
                else:
                    normed_rewards[k] = [(np.array(v) - mean)/std for v in arr]

        # 加和，再整体归一化（如有需要）
        if reward_weights is None:
            summed = [sum(normed_rewards[k][i] for k in reward_keys) for i in range(len(group))]
        else:
            reward_weight_sum = sum(reward_weights[k] for k in reward_keys)
            summed = [sum(normed_rewards[k][i] * reward_weights[k]/reward_weight_sum for k in reward_keys) for i in range(len(group))]

        for idx, episode in enumerate(group):
            summed_norm = summed[idx].tolist()
            if isinstance(summed_norm, float):
                summed_norm = [summed_norm] * len(episode.generated_token_ids)
            # 更新归一化后的各项（如要保留）
            new_reward_info = episode.reward_info.copy()
            for k in reward_keys:
                new_reward_info[k] = normed_rewards[k][idx]
            # 新reward可以是 summed[idx] 或 summed_norm，视实验需求
            # print(summed_norm)
            episode_new = dataclasses.replace(episode, reward=summed_norm, reward_info=new_reward_info)
            output.append(episode_new)
    return output, has_grad



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def compute_kl_loss(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k3",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio

