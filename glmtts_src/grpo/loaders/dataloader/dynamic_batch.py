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
import logging
import random
import time

from torch.utils.data import IterableDataset

# from .iterable import MapperIterDataPipe


# class MapperIterDataPipe(IterableDataset):
class PipeWrapper(IterableDataset):

    def __init__(self, datapipe, fn):
        self.datapipe = datapipe
        self.fn = fn

    # def set_epoch(self, epoch):
    #     self.epoch = epoch

    def __iter__(self):
        assert callable(self.fn)
        for data in self.datapipe:
            if data is not None:
                yield self.fn(data), data


# class MaxTokenBucketizerIterDataPipe(IterableDataset):
class BucketizerPipe(IterableDataset):

    def __init__(
            self,
            datapipe,
            len_fn,
            batch_size,
            buffer_size=0,  # 10240,
            bucket_size=500,
            # batch_mode="padding",
    ):
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        assert buffer_size >= 0, "Buffer size is required to be larger than 0!"
        assert bucket_size > 0, "Bucket size is required to be larger than 0!"

        datapipe = PipeWrapper(datapipe, fn=len_fn)
        self.len_fn = len_fn
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.bucket_size = bucket_size
        # self.batch_mode = batch_mode

    # def set_epoch(self, epoch):
    #     self.epoch = epoch

    def __iter__(self):
        buffer = []
        batch = []
        bucket = []
        # max_lengths = 0
        # min_lengths = 999999
        # batch_lengths = 0

        if self.buffer_size == 0:  # 不buffer, 直接bucket (buffer只是为了shuffle)
            _cnt, N = 0, 10
            tic = time.perf_counter()
            for sample in self.datapipe:
                bucket.append(sample)
                # 3. split bucket to batches
                if len(bucket) == self.bucket_size:
                    _dur = time.perf_counter() - tic
                    for _batch in self.bucket2batches(bucket, batch):
                        yield _batch
                        _cnt += 1
                    logging.debug(f"BucketizerPipe-{self.datapipe.datapipe.meta_gen.worker_id}: sec/batch {_dur} / {_cnt} = {_dur/_cnt}")
                    tic = time.perf_counter()
                    _cnt = 0

            if bucket:  # bucket size过大, 装下了所有buffer
                for _batch in self.bucket2batches(bucket, batch):
                    yield _batch

            if batch:  # batch size过大, 装下了所有bucket
                yield batch
        else:
            assert self.buffer_size > 0
            for d in self.datapipe:
                # 1. add samples to buffer
                if d[0] > self.batch_size:
                    continue
                buffer.append(d)
                # 2. split buffer to buckets
                if len(buffer) == self.buffer_size:
                    for _batch in self.buffer2batches(buffer, bucket, batch):
                        yield _batch

            if buffer:  # datapipe结束后, buffer还有剩余 (buffer size过大)
                for _batch in self.buffer2batches(buffer, bucket, batch):
                    yield _batch

            if bucket:  # bucket size过大, 装下了所有buffer
                for _batch in self.bucket2batches(bucket, batch):
                    yield _batch

            if batch:  # batch size过大, 装下了所有bucket
                yield batch
                batch.clear()

    def buffer2batches(self, buffer, bucket, batch):
        random.shuffle(buffer)
        for sample in buffer:
            bucket.append(sample)
            # 3. split bucket to batches
            if len(bucket) == self.bucket_size:
                for _batch in self.bucket2batches(bucket, batch):
                    yield _batch
        # buffer = []
        buffer.clear()

    def bucket2batches(self, bucket, batch):
        bucket.sort(key=lambda d: d[0])
        if batch:  # next bucket, prev batch (not full)
            max_lengths = max([self.len_fn(x) for x in batch])
        else:
            max_lengths = 0

        for x in bucket:
            # length, _, token = x
            length, token = x
            if length > max_lengths:
                max_lengths = length
            # 把当前sample放入batch后的token总数
            batch_lengths = max_lengths * (len(batch) + 1)
            if batch_lengths > self.batch_size:
                yield batch
                # batch = []
                batch.clear()
                max_lengths = length
            batch.append(token)
        # bucket = []
        bucket.clear()
