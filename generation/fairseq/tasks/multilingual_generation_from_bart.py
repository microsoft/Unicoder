# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import logging
import itertools
import numpy as np

from fairseq.data import GenerationMultiPairDataset 
from fairseq import options, utils

from .translation import TranslationTask
from . import register_task

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    SortDataset,
    data_utils,
    encoders,
    indexed_dataset,
    ResamplingDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)


logger = logging.getLogger(__name__)


def load_generation_pair_dataset(
    data_path, split,
    tgt,
    src_dict,
    tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False, common_eos=None,
    lg_id=None
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, "src", "tgt", "src", data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, "src", "tgt"))
        elif split_exists(split_k, "tgt", "src", "src", data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, "tgt", "src"))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + "src", src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + "tgt", tgt_dict, dataset_impl)

        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, "src", "tgt", len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, lg_id)
        if common_eos is not None:
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(common_eos)))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(common_eos)))
            eos = tgt_dict.index('[{}]'.format(common_eos))

    bos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, "src", "tgt"))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return GenerationMultiPairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos, bos=bos 
    )


@register_task('multilingual_generation_from_bart')
class MultilingualGenerationFromBARTTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, for example, "en,de,fr"'
                                 'be careful these langs are what you used for pretraining (the same order),'
                                 'not for finetuning.'
                                 'you should always add all pretraining language idx during finetuning.')
        parser.add_argument('--multilang-sampling-alpha', type=float, default=0.7,
                            help='sub sampling factor')
        parser.add_argument('--common_eos', type=str,
                            help='common eos symbol for all languages')
        parser.add_argument('--placeholder', type=int, default=0,
                            help='number of placeholder in dictionaries')
        parser.add_argument('--gt-langs', type=str,
                            help="languages used in generation finetuning, separated wiht -, for example, 'en-fr-de'")

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(',')
        for d in [src_dict, tgt_dict]:
            d.add_symbol('<mask>')
            if args.common_eos is not None:
                d.add_symbol('[{}]'.format(args.common_eos))
            for l in self.langs:
                d.add_symbol('[{}]'.format(l))
            for i in range(args.placeholder):
                d.add_symbol('[placeholder_{}]'.format(i))

        if args.gt_langs is None:
            self.gt_langs = [args.source_lang]
        else:
            self.gt_langs = args.gt_langs.split('-')

        assert len(self.gt_langs) > 0
        for lg in self.gt_langs:
            assert lg in self.langs

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], args.source_lang, 'dict.src.txt'))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], args.target_lang, 'dict.tgt.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        
        lg_datasets = []
        for lg in self.gt_langs:
            src, tgt = lg, lg 
            bos_id = self.tgt_dict.index('[{}]'.format(lg))
            data_path_lg = os.path.join(data_path, lg)
            dataset = load_generation_pair_dataset(
                data_path_lg, split, tgt, self.src_dict, self.tgt_dict,
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=getattr(self.args, 'max_source_positions', 1024),
                max_target_positions=getattr(self.args, 'max_target_positions', 1024),
                load_alignments=self.args.load_alignments,
                prepend_bos=getattr(self.args, 'preprend_bos', False),
                append_source_id=True,
                common_eos=self.args.common_eos,
                lg_id=bos_id
                )
            lg_datasets.append(dataset)
        
        dataset_lengths = np.array([len(d) for d in lg_datasets], dtype=float) 

        sample_probs = self._get_sample_prob(dataset_lengths)
        logger.info("| Sample probability by language: ", {
                lang: "{0:.4f}".format(sample_probs[id])
                for id, lang in enumerate(self.gt_langs)
            }
        )
        size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
        logger.info("| Up/Down Sampling ratio by language: ", {
                lang: "{0:.2f}".format(size_ratio[id])
                for id, lang in enumerate(self.gt_langs)
            }
        )
        if split == getattr(self.args, "train_subset", "train"):
            resampled_lang_datasets = [
                ResamplingDataset(
                    lg_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lg_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
                )
        else:
            dataset = ConcatDataset(lg_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lg_datasets):
                split_name = split + '_' + self.gt_langs[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset
            
            if hasattr(self.args, "valid_subset"):
                if split in self.args.valid_subset:
                    self.args.valid_subset = self.args.valid_subset.replace(
                        split, ','.join(lang_splits)
                    )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))
        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        ) 

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            bos_token = self.tgt_dict.index('[{}]'.format(self.args.target_lang))
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token)

    def build_generator(self, models, args):
        tgt_lang = self.args.target_lang if self.args.common_eos is None else self.args.common_eos
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index('[{}]'.format(tgt_lang))
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                eos=self.tgt_dict.index('[{}]'.format(tgt_lang))
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang = self.args.source_lang if self.args.common_eos is None else self.args.common_eos

        src_lang_id = self.source_dictionary.index('[{}]'.format(src_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)
        return dataset
