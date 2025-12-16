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
import itertools
import copy
import time

import torch


def create(block_list, tensor_len):
    assert type(block_list) == list
    assert len(block_list) > 0
    assert tensor_len > 0

    block_list = copy.deepcopy(block_list)

    block_list = block_list + [block_list[-1]] * 10000

    block_list_accumulate = list(itertools.accumulate(block_list, lambda x, y: x + y))

    zeros = torch.zeros([tensor_len, tensor_len])

    for i in range(tensor_len):
        for j in range(len(block_list_accumulate)):
            if i < block_list_accumulate[j]:
                cur_block_size = block_list[j]

                if j == 0:
                    last_size = 0
                else:
                    last_size = block_list_accumulate[j - 1]

                break

        index_in_cur_block = i - last_size
        look_future_size = cur_block_size - 1 - index_in_cur_block

        if cur_block_size <= 172:
            min_future_size = int(cur_block_size / 2)
        else:
            min_future_size = int(cur_block_size / 1.5)

        if look_future_size >= min_future_size:
            delta = 0
        else:
            delta = min_future_size - look_future_size

        aim_len_with_accumu = last_size + cur_block_size + delta
        zeros[i, 0:aim_len_with_accumu] = 1

    return zeros.int()


global_cache = {}


def create_with_cache(block_list, tensor_len):
    key = ",".join([str(i) for i in block_list])
    if key not in global_cache:
        global_cache[key] = create(block_list, 10000)
    big_tensor = global_cache[key]
    assert big_tensor.shape[0] > tensor_len
    result = big_tensor[0:tensor_len, 0:tensor_len].clone()
    return result


if __name__ == "__main__":
    t1 = time.time()
    res = create_with_cache([2, 4, 8], 20)
    t2 = time.time()
    res = create_with_cache([2, 4, 8], 40)
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)