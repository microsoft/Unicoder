# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
XLM-K: Improving Cross-Lingual Language Model Pre-Training with Multilingual Knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryEfficientGather(object):
    def __init__(self, world_size):
        self.tensors_gather = None
        self.world_size = world_size

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if self.world_size == 1:
            return tensor
        if self.tensors_gather is None:
            self.tensors_gather = [torch.ones_like(tensor)
                                   for _ in range(self.world_size)]

        torch.distributed.all_gather(self.tensors_gather, tensor, async_op=False)
        return torch.cat(self.tensors_gather, dim=0)


class MultiTaskGather(object):
    def __init__(self, batch_size, hidden_size, distributed_world_size):
        self.distributed_world_size = distributed_world_size
        self.queue_gather = MemoryEfficientGather(self.distributed_world_size)
        self.queue_id_gather = MemoryEfficientGather(self.distributed_world_size)
        self.task_id_gather = MemoryEfficientGather(self.distributed_world_size)

        self.queue = torch.zeros(batch_size, hidden_size).cuda().half()
        self.task_id = torch.ones(batch_size, dtype=torch.long).cuda() * -1
        self.queue_id = torch.ones(batch_size, dtype=torch.long).cuda() * -1

    def clear(self):
        self.queue.fill_(0.0)
        self.task_id.fill_(-1)
        self.queue_id.fill_(-1)

    def gather(self):
        queue = self.queue_gather.concat_all_gather(self.queue)
        task_id = self.task_id_gather.concat_all_gather(self.task_id)
        queue_id = self.queue_id_gather.concat_all_gather(self.queue_id)

        task2tensors = {}
        tasks = set(task_id.tolist())
        for task in tasks:
            task2tensors[task] = {}
            task2tensors[task]['queue'] = queue[task_id == task]
            task2tensors[task]['queue_id'] = queue_id[task_id == task]

        return task2tensors


class EmbeddingQueue(object):
    def __init__(self, queue_size, dim, distributed_world_size, fp16=False, multi_queue=False):
        self.distributed_world_size = distributed_world_size
        self.queue_size = queue_size

        self.queue = (torch.randn((queue_size, dim)) / (math.sqrt(dim))).cuda()
        if fp16:
            self.queue = self.queue.half()
        if multi_queue:
            self.queue_id = torch.ones(queue_size, dtype=torch.long).cuda() * -1
            print("queue id shape", self.queue_id.shape)
        else:
            self.queue_id = None

        self.random_sample_number = queue_size

    def push(self, tensor, queue_id=None):
        if self.queue_id is not None:
            assert queue_id is not None

        if len(tensor) == 0:
            return

        sample_number = tensor.shape[0]
        self.random_sample_number = max(self.random_sample_number - sample_number, 0)

        self.queue.data = torch.cat((self.queue[sample_number:].data, tensor.data), 0)

        if self.queue_id is not None:
            assert queue_id.shape[0] == sample_number, "query id number = {0}, sample number = {1}".format(
                queue_id.shape[0], sample_number)
            self.queue_id.data = torch.cat((self.queue_id[sample_number:].data, queue_id.data), 0)

        if self.queue.shape[0] > self.queue_size:
            assert tensor.shape[0] > self.queue_size
            idx = torch.randperm(self.queue.shape[0])[:self.queue_size]
            self.queue = self.queue[idx]
            if self.queue_id is not None:
                self.queue_id = self.queue_id[idx]


class ProjectionHead(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, layer, args):
        super().__init__()
        self.layer = layer
        self.projection_list = torch.nn.ModuleList()
        self.batch_norm_list = torch.nn.ModuleList()
        self.n_lg = 1
        if args.sa_lg_bn:
            self.n_lg = 20
        if self.layer >= 1:
            for i in range(self.layer):
                layer_in_dim = mid_dim if (i != 0) else in_dim
                layer_out_dim = mid_dim if (i != self.layer-1) else out_dim
                self.projection_list.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=False))

                affine = True if (i != self.layer-1) else False
                if not args.sync_bn:
                    for _ in range(self.n_lg):
                        self.batch_norm_list.append(torch.nn.BatchNorm1d(layer_out_dim, eps=1e-12, affine=affine))
                else:
                    self.batch_norm_list.append(torch.nn.SyncBatchNorm(layer_out_dim, eps=1e-12, affine=affine))

    def forward(self, tensor, lg_id=0):
        for i in range(self.layer):
            tensor = self.projection_list[i](tensor)
            tensor = self.batch_norm_list[i * self.n_lg + lg_id % self.n_lg](tensor)
            if i != self.layer - 1:
                tensor = F.relu(tensor)
        return tensor


class ContrastiveLossWithQueue(object):
    def __init__(self, queue_size, dim, args, multi_queue=False):
        self.args = args
        self.multi_queue = multi_queue
        assert self.args.enable_memory_bank
        self.queue = EmbeddingQueue(
            queue_size=queue_size, dim=dim,
            distributed_world_size=self.args.distributed_world_size,
            fp16=self.args.fp16, multi_queue=multi_queue
        )

    def masked_loss(self, tensor_query, concated_tensor, query_id, key_id, labels, mask):
        if self.multi_queue:
            if self.queue is not None:
                concated_queue_id = torch.cat([query_id, key_id, self.queue.queue_id])
            else:
                concated_queue_id = torch.cat([query_id, key_id])
            different_queue_id = (query_id.unsqueeze(1) != concated_queue_id.unsqueeze(0))
            if self.queue.random_sample_number > 0:
                different_queue_id = different_queue_id & (concated_queue_id.unsqueeze(0) != -1)
            mask = mask | different_queue_id

        logits = torch.matmul(tensor_query, concated_tensor.T) / self.args.cts_temp
        logits.masked_fill_(mask, float('-inf'))
        loss = F.nll_loss(
            F.log_softmax(
                logits,
                dim=-1,
                dtype=torch.float32,
            ),
            labels,
            reduction='sum',
        )
        predict = torch.argmax(logits.float(), dim=1)
        accuracy = torch.sum(predict == labels).item() / labels.size()[0]
        self.accuracy = accuracy
        return loss

    def get_loss(self, tensor_query, tensor_key, mask=None, query_id=None, key_id=None, bi_direction=False):
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = (mask == 1)
            if not mask.any():
                return torch.tensor(0.0), 0
            tensor_query = tensor_query[mask]
            tensor_key = tensor_key[mask]
            if self.multi_queue:
                assert query_id is not None and key_id is not None
                query_id = query_id[mask]
                key_id = key_id[mask]

        sample_number = tensor_query.shape[0]
        assert tensor_key.shape[0] == sample_number

        tensor_queue = self.queue.queue
        if self.args.enable_l2_norm:
            tensor_query = F.normalize(tensor_query, dim=-1, eps=1e-4)
            tensor_key = F.normalize(tensor_key, dim=-1, eps=1e-4)
            tensor_queue = F.normalize(tensor_queue, dim=-1, eps=1e-4)

        if bi_direction:
            mask = torch.eye(sample_number, m=sample_number*2 + tensor_queue.shape[0], dtype=torch.bool).cuda()
            labels = torch.arange(sample_number, dtype=torch.long).cuda() + sample_number
            acc = 0
            loss_forward = self.masked_loss(tensor_query, torch.cat([tensor_query, tensor_key, tensor_queue]), query_id,
                                            key_id, labels, mask)
            acc += self.accuracy
            loss_backward = self.masked_loss(tensor_key, torch.cat([tensor_key, tensor_query, tensor_queue]), key_id,
                                            query_id, labels, mask)
            acc += self.accuracy
            self.accuracy = acc / 2
            loss = loss_forward + loss_backward
            sample_number *= 2
        else:
            mask = torch.zeros((sample_number, sample_number+tensor_queue.shape[0]), dtype=torch.bool).cuda()
            labels = torch.arange(sample_number, dtype=torch.long).cuda()
            loss = self.masked_loss(tensor_query, torch.cat([tensor_key, tensor_queue]), query_id, key_id, labels, mask)
        return loss, sample_number
