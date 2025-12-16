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
import os
from pathlib import Path
import random

def get_syn_text():
    lines = open('grpo/data/sample_syn_text.txt').readlines()
    syn_texts = []
    for idx, line in enumerate(lines):
        emo_tag = 6
        text_id = 'sample-6-'+str(idx)
        syn_texts.append((text_id, line.strip(), emo_tag))
    return syn_texts
    
def split_text(syn_texts):
    from cosyvoice.utils.frontend_utils import split_hard, split_into_min_sentence
    
    split_sentences = []
    for uttid, syn_text, emotion in syn_texts:
        min_sentences, at_least_one_sentence = split_into_min_sentence(syn_text, 30)
        sentence_x_units = split_hard(min_sentences, 60)
        res = [''.join(units) for units in sentence_x_units]
        res = [i for i in res if len(i)>4]
        # split_sentences.extend(res)
        for i, sentence in enumerate(res):
            split_sentences.append((uttid+'_'+str(i).zfill(2), sentence, emotion))
    return split_sentences
        

syn_texts = get_syn_text()

syn_texts = split_text(syn_texts)
# 把所有item按顺序收集到一个大列表
x = [
    'examples/corner.jsonl'
]

all_items = []

for meta_path in x:
    with open(meta_path, "r", encoding="utf-8") as fin:
        feat_root = None
        for idx, line in enumerate(fin):
            item = json.loads(line)
            all_items.append(item)

item_num = len(all_items)
syn_num = len(syn_texts)
random.shuffle(syn_texts)
random.shuffle(all_items)




feat_root = '/your/root/'


write_num = 0
for i, syn_text_item in enumerate(syn_texts):
    if write_num % 1000 == 0:
        out_jsonl = f'grpo/data/{i//1000}.jsonl'
        fout = open(out_jsonl, "w", encoding="utf-8")
    item = all_items[i % item_num]  # 循环利用item
    uttid, syn_text, emotion = syn_text_item
    prompt_uttid = item['prompt_speech'].split('/')[-1].split('.')[0]
    new_id = prompt_uttid + '--' + uttid
    write_num += 1
    new_item = {
        "uttid": prompt_uttid + '--' + uttid,
        "prompt_text": item["prompt_text"],
        "prompt_speech": os.path.join(feat_root, 'waveform', item["prompt_speech"].split('/')[-1]),
        "prompt_speech_token": os.path.join(feat_root, "prompt_speech_token", prompt_uttid + '.npy'),
        "embedding": os.path.join(feat_root, "embedding", prompt_uttid + '.npy'),
        "prompt_speech_feat": os.path.join(feat_root, "prompt_speech_feat", prompt_uttid + '.npy'),
        "syn_text": syn_text,
        "emotion": emotion,
    }
    fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
    

print(f"共输出数据 {syn_num} 条, item池大小 {item_num}")
print(f"输出文件: {out_jsonl}")