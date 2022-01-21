# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
XLM-K: Improving Cross-Lingual Language Model Pre-Training with Multilingual Knowledge
"""

import math
from numpy.core.numeric import indices

import torch
import torch.nn.functional as F

from fairseq import utils,tasks

from fairseq.data import Dictionary

from . import FairseqCriterion, register_criterion
from fairseq.modules import ContrastiveLossWithQueue, MultiTaskGather
from fairseq.tasks.xlmk import UnicoderTaskSpace
import pickle
import os
from enum import Enum, auto
import random

class UnicoderLossSpace(Enum):
    mlm = auto()
    wd_mlm = auto()
    wd_w2d = auto()
    wd_d2w = auto()


@register_criterion('xlmk_loss')
class XLMKLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        self.dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))

        self.contrastive_loss_dic = {}
        if self.args.enable_word2doc or self.args.enable_doc2word:
            loss_dic = {}
            if self.args.enable_doc2word:
                loss_dic['doc2word'] = ContrastiveLossWithQueue(
                    queue_size=self.args.memory_bank_size,
                    dim=self.args.encoder_embed_dim,
                    args=self.args,
                    multi_queue=False
                )
            if self.args.enable_word2doc:
                loss_dic['word2doc'] = ContrastiveLossWithQueue(
                    queue_size=self.args.memory_bank_size,
                    dim=self.args.encoder_embed_dim,
                    args=self.args,
                    multi_queue=False
                )
            self.contrastive_loss_dic[UnicoderTaskSpace.wiki_definition] = loss_dic

        if len(self.contrastive_loss_dic) > 0:
            contrastive_max_sentence = self.args.max_sentences if self.args.contrastive_max_sentence == -1 \
                else self.args.contrastive_max_sentence
            self.variable_0_gather = MultiTaskGather(
                batch_size=contrastive_max_sentence,
                hidden_size=self.args.encoder_embed_dim,
                distributed_world_size=self.args.distributed_world_size
            )
            self.variable_1_gather = MultiTaskGather(
                batch_size=contrastive_max_sentence,
                hidden_size=self.args.encoder_embed_dim,
                distributed_world_size=self.args.distributed_world_size
            )
        self.src_queue = None
        self.tgt_queue = None
        self.eval_state = 0

    def update_variable(self, extra_src, extra_tgt, src_lg):
        self.variable_0_gather.queue.data = extra_src['contrastive_state'].data
        self.variable_1_gather.queue.data = extra_tgt['contrastive_state'].data

        if src_lg is not None:
            self.variable_0_gather.queue_id.data = src_lg.data
            self.variable_1_gather.queue_id.data = src_lg.data


    def forward(self, model, sample, reduce=True):

        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss_dic = {}
        sample_size_dic = {}
        for task in list(UnicoderLossSpace):
            loss_dic[task] = torch.tensor(0.0)
            sample_size_dic[task] = 0

        def update_loss_dic(loss_enum, loss, sample_size):
            loss_dic[loss_enum] = loss
            sample_size_dic[loss_enum] = sample_size

        task = UnicoderTaskSpace(sample['net_input']['task'][0].item())
        current_batch_task = task
        all_task = sample['net_input']['task'].tolist()
        assert all([item == task.value for item in all_task]), str(all_task)
        if len(self.contrastive_loss_dic) > 0:
            self.variable_1_gather.clear()
            self.variable_0_gather.clear()
        if task == UnicoderTaskSpace.mlm:
            targets = sample['targets'][task.name]['target']
            masked_tokens = targets.ne(self.padding_idx)

            sample_size = masked_tokens.int().sum().item()
            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                masked_tokens = None
            assert sample['net_input'][task.name]['src_tokens'].shape[1] < 513    
            _, logits, extra = model(src_tokens=sample['net_input'][task.name]['src_tokens'],
                                  task=task,
                                  masked_tokens=masked_tokens,
                                  return_all_hiddens=True)
            if sample_size != 0:
                targets = targets[masked_tokens]
            loss = F.nll_loss(
                F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            update_loss_dic(UnicoderLossSpace.mlm, loss, sample_size)
        elif task == UnicoderTaskSpace.wiki_definition:
            targets = sample['targets'][task.name]['target']
            masked_tokens = targets.ne(self.padding_idx)
            sample_size = masked_tokens.int().sum().item()
            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                masked_tokens = None

            if sample_size != 0:
                targets = targets[masked_tokens]

            _, logits_word, extra_word = model(
                src_tokens=sample['net_input'][task.name]['src_tokens'],
                task=task,
                return_all_hiddens=True,
                masked_tokens=masked_tokens,
                start_end_position=sample['net_input'][task.name]['start_end_position'],
                variable_index=0,
            )
            _, logits_doc, extra_doc = model(
                src_tokens=sample['net_input'][task.name]['document_tokens'],
                task=task,
                return_all_hiddens=True,
                variable_index=1,
            )

            self.update_variable(extra_word, extra_doc, None)

            mask = sample['net_input'][task.name]['start_end_position'][:, 0] != -1
            if self.args.enable_word2doc:
                loss_word2doc, sample_size_word2doc = self.contrastive_loss_dic[task]['word2doc'].get_loss(
                    tensor_query=extra_word['contrastive_state'],
                    tensor_key=extra_doc['contrastive_state'],
                    query_id=None,
                    key_id=None,
                    bi_direction=True,
                    mask=mask
                )
                update_loss_dic(UnicoderLossSpace.wd_w2d, loss_word2doc, sample_size_word2doc)
            else:
                loss_word2doc = torch.tensor(0.0)
                sample_size_word2doc = 0
            if self.args.enable_doc2word:
                loss_doc2word, sample_size_doc2word = self.contrastive_loss_dic[task]['doc2word'].get_loss(
                    tensor_query=extra_doc['contrastive_state'],
                    tensor_key=extra_word['contrastive_state'],
                    query_id=None,
                    key_id=None,
                    bi_direction=True,
                    mask=mask
                )
                update_loss_dic(UnicoderLossSpace.wd_d2w, loss_doc2word, sample_size_doc2word)
            else:
                loss_doc2word = torch.tensor(0.0)
                sample_size_doc2word = 0

            loss_mlm = F.nll_loss(
                F.log_softmax(logits_word.view(-1, logits_word.size(-1)), dim=-1, dtype=torch.float32),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            update_loss_dic(UnicoderLossSpace.wd_mlm, loss_mlm, sample_size)
            word2doc_ratio = sample_size / sample_size_word2doc if sample_size_word2doc > 0 else 0
            doc2word_ratio = sample_size / sample_size_doc2word if sample_size_doc2word > 0 else 0
            loss = loss_mlm + \
                   loss_word2doc * word2doc_ratio * self.args.word2doc_scale + \
                   loss_doc2word * doc2word_ratio * self.args.doc2word_scale
        else:
            raise ValueError("unknown task", task)

        if len(self.contrastive_loss_dic) > 0:
            self.variable_0_gather.task_id.data.fill_(task.value)
            self.variable_1_gather.task_id.data.fill_(task.value)
            variable_0_task2tensors = self.variable_0_gather.gather()
            variable_1_task2tensors = self.variable_1_gather.gather()

            for task_id in variable_0_task2tensors.keys():
                if task_id < 0:
                    continue
                task = UnicoderTaskSpace(task_id)
                if task not in self.contrastive_loss_dic:
                    continue
                contrastive_losses = self.contrastive_loss_dic[task]
                if isinstance(contrastive_losses, dict):
                    if task == UnicoderTaskSpace.wiki_definition:
                        if self.args.enable_word2doc:
                            contrastive_losses['word2doc'].queue.push(
                                variable_1_task2tensors[task_id]['queue'],
                                variable_1_task2tensors[task_id]['queue_id'])
                        if self.args.enable_doc2word:
                            contrastive_losses['doc2word'].queue.push(
                                variable_0_task2tensors[task_id]['queue'],
                                variable_0_task2tensors[task_id]['queue_id'])
                    else:
                        raise NotImplementedError("Unknown task in push tensor to loss queue: {0}".format(task))
                else:
                    contrastive_losses.queue.push(variable_0_task2tensors[task_id]['queue'],
                                                  variable_0_task2tensors[task_id]['queue_id'])
                    contrastive_losses.queue.push(variable_1_task2tensors[task_id]['queue'],
                                                  variable_1_task2tensors[task_id]['queue_id'])
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'indices' : torch.zeros(0) if not self.args.wa_need_id else indices,
        }
        for task in UnicoderLossSpace:
            logging_output['loss_' + task.name] = utils.item(loss_dic[task].data) if hasattr(loss_dic[task], 'data') else loss_dic[task]
            logging_output['sample_size_' + task.name] = sample_size_dic[task]
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # indices = None
        indices = torch.cat(tuple(log.get('indices', torch.zeros(0)).reshape(-1).detach().cpu() for log in logging_outputs))

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'indices': indices
        }
        for task in UnicoderLossSpace:
            sample_size_task = sum(log.get('sample_size_' + task.name, 0) for log in logging_outputs)
            loss_task = sum(log.get('loss_' + task.name, 0) for log in logging_outputs) / sample_size_task / math.log(2) \
                if sample_size_task > 0 else 0.

            if 'acc' in task.name or 'recall' in task.name or 'count' in task.name:
                loss_task = sum(
                    log.get('loss_' + task.name, 0) for log in logging_outputs) / len(logging_outputs) \
                    if sample_size_task > 0 else 0.
                if 'acc' in task.name or 'recall' in task.name:
                    loss_task *= 100

            agg_output['loss_' + task.name] = loss_task

        return agg_output

