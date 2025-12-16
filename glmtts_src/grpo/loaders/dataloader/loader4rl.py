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
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset
from time import perf_counter
from .feature_loader_rl import FeatureLoader

SEED = 10086  # 固定种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def iter_jsonl(fp):
    root = None

    try:
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                line_item = json.loads(line.strip())
                yield line_item
            
    except Exception as e:
        logging.error(f'Error loading jsonl file {fp}: {e}')
        yield []
        
        

def count_line_num(fp):
    import os
    size = os.path.getsize(fp) / 2 * 20  # MB
    # with open(fp, 'r') as f:
    #     return sum([1 for _ in f])
    return size  # 计算行数太慢了, 直接用文件大小


# class JsonDataset:
class MetaReader:
    """
    多个jsonl作为source
    有一个buffer, 缓存jsonl里的line (sample meta item)
    随机从各个jsonl src里load line到buffer内, 随机的概率为jsonl的行数比例
    """

    def __init__(self, json_files, buffer_size=5000):
        super(MetaReader).__init__()
        assert len(json_files) > 0, 'There should be >=1 json_files'
        # sorted(list(set(['/'.join(e.split('/')[:7]) for e in json_files])))  # check dataset lists
        self.all_json_files = json_files
        self.json_files = None  # 等待worker init fn分配
        self.buffer_size = buffer_size
        self.data_pool = []

        self.worker_id = None
        # self.num_worker = None
        self.initialized = False

    def worker_init(self, start, end, worker_id):
        """
        每次创建DataLoader时调用, 将`self.all_json_files`均分成num_worker个split, 然后再根据worker_id取出自己的部分, 设为`self.all_json_files`
        """
        self.json_files = self.all_json_files[start:end]
        assert len(self.json_files) > 0
        self.worker_id = worker_id

    # def init(self):
    def reset(self):
        if self.json_files is None:
            logging.warning('MetaReader work in single process mode.')
            self.json_files = self.all_json_files

        logging.debug(f'worker {self.worker_id} reseting: {len(self.json_files)=}.')
        self.json_iterators = [self._data_streamer(file) for file in self.json_files]
        self.json_file_weights = [count_line_num(file) for file in self.json_files]
        self._fill_buffer_until_full()

    def _data_streamer(self, json_file):
        yield from iter_jsonl(json_file)

    # def _fill_buffer(self):  # random
    #     succeed = False
    #     while len(self.json_iterators) > 0:
    #         _indexes = list(range(len(self.json_iterators)))
    #         _idx = random.choices(_indexes, weights=self.json_file_weights, k=1)[0]
    #         json_iterator = self.json_iterators[_idx]
    #         try:
    #             item = next(json_iterator)
    #             self.data_pool.append(item)
    #             succeed = True
    #             break
    #         except StopIteration:
    #             # 删除结束的迭代器并重新计算剩余文件的权重
    #             self.json_iterators.pop(_idx)
    #             self.json_file_weights.pop(_idx)
    #     return succeed  # False: out of data

    def _fill_buffer(self):
        succeed = False
        while len(self.json_iterators) > 0:
            _idx = 0
            json_iterator = self.json_iterators[_idx]
            try:
                item = next(json_iterator)
                self.data_pool.append(item)
                succeed = True
                break
            except StopIteration:
                # 删除结束的迭代器并重新计算剩余文件的权重
                self.json_iterators.pop(_idx)
                self.json_file_weights.pop(_idx)
        return succeed  # False: out of data

    def _fill_buffer_until_full(self):
        succeed = True
        while succeed and (len(self.data_pool) < self.buffer_size):
            succeed = self._fill_buffer()  # if out of data , succeed is False

    def __iter__(self):
        self.reset()  # 必须放在DataLoader开启多进程之后, 在子进程里reset

        logging.debug(f'worker {self.worker_id} MetaReader __iter__; {len(self.json_iterators)=}')
        #if len(self.json_iterators) == 0:
        #    import pdb
        #    pdb.set_trace()
        while len(self.data_pool) > 0:
            item = self.data_pool.pop(random.randint(0, len(self.data_pool) - 1))
            self._fill_buffer()  # 每次 pop 后，都填充 buffer
            yield item

        logging.debug(f'worker {self.worker_id} finished.')


class Loader(IterableDataset):
    """
    root
    ---
    ["01ad87609b26797deec730e282d0a2f6_0017_00020", {"acoustic_token": "acoustic_token/xiaoyuzhou_part2_00000.tar/01ad87609b26797deec730e282d0a2f6_0017_00020.npy", "info_ASRv1": "info_ASRv1/xiaoyuzhou_part2_00000.tar/01ad87609b26797deec730e282d0a2f6_0017_00020.json", "phone_id_ASRv1": "phone_id_ASRv1/xiaoyuzhou_part2_00000.tar/01ad87609b26797deec730e282d0a2f6_0017_00020.npy", "semantic_token": "semantic_token/xiaoyuzhou_part2_00000.tar/01ad87609b26797deec730e282d0a2f6_0017_00020.npy", "wav24khz": "wav24khz/xiaoyuzhou_part2_00000.tar/01ad87609b26797deec730e282d0a2f6_0017_00020.wav"}]
    """

    def __init__(self, meta_pattern, feature_loader, max_epoch=0):
        # self.meta_fps = sorted(glob.glob(meta_pattern))
        self.feature_loader: FeatureLoader = feature_loader
        self.max_epoch = max_epoch
        self.ep = 0

        json_files = []
        if isinstance(meta_pattern, str):
            meta_pattern = []
        for e in meta_pattern:
            _files = sorted(glob.glob(e))
            assert len(_files) > 0, e
            json_files.extend(_files)

        self.meta_gen = MetaReader(json_files)

    def __iter__(self):
        tic = perf_counter()
        cnt, N = 0, 100
        while self.max_epoch <= 0 or self.ep < self.max_epoch:
            for d_item in self.meta_gen:
                # d = {'uttid': d_item.pop('uttid')}
                # for feat_name, (tar_filepath, sample_nm) in d_item.items():
                # for feat_name in self.feature_loader.feature_keys:
                #     tar_filepath, sample_nm = d_item[feat_name]
                #     tar_reader = self.tars.setdefault(tar_filepath, TarReader(tar_filepath))
                #     sample_bytes = tar_reader.get(sample_nm)
                #     d[feat_name] = sample_bytes

                loaded_data = self.feature_loader(d_item)

                # if self.max_tar_stream >= 0:  # restrict the opened stream num
                #     while len(self.tars) > self.max_tar_stream:
                #         _tar_fp, _tar_reader = self.tars.popitem(last=False)
                #         del _tar_reader

                yield loaded_data
                cnt += 1
                if cnt % N == 0:
                    _dur = perf_counter() - tic
                    logging.debug(f'Loader-{self.meta_gen.worker_id}: sec/sample {_dur} / {cnt} = {_dur/cnt}')
                    tic = perf_counter()
                    cnt = 0
            self.ep += 1
            logging.debug(f'Loader-{self.meta_gen.worker_id} finished ep {self.ep}')
