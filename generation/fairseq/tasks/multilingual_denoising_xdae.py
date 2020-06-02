# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import glob

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    AppendTokenDataset,
    ConcatDataset,
    XDAEDenoisingDataset,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
)
from .denoising import DenoisingTask
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


@register_task('xdae_multilingual_denoising')
class XDAEMultilingualDenoisingTask(DenoisingTask):

    @staticmethod
    def add_args(parser):
        DenoisingTask.add_args(parser)
        parser.add_argument('--multilang-sampling-alpha', type=float, default=1.0,
                            help='smoothing alpha for sample rations across multiple datasets')
        parser.add_argument('--add-lang-token', default=False, action='store_true')
        parser.add_argument('--no-prepend-sent-bos', default=False, action='store_true')

        parser.add_argument('--common-eos', type=str, 
                            help="add common eos to samples")
        parser.add_argument('--add-placeholder', type=int, default=-1,
                            help="placeholder for more special ids such as language ids")

        parser.add_argument('--word-shuffle', type=float, default=0,
                            help="Randomly shuffle input words (0 to disable)")
        parser.add_argument("--word-dropout", type=float, default=0,
                            help="Randomly dropout input words (0 to disable)")
        parser.add_argument("--word-blank", type=float, default=0,
                            help="Randomly blank input words (0 to disable)")

        parser.add_argument('--sampled-data', default=False, action='store_true')
        parser.add_argument('--langs', type=str, help="language ids we are considering", default=None)
        parser.add_argument('--no-whole-word-mask-langs', type=str, default='', metavar='N',
                            help='languages without spacing between words dont support whole word masking')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        paths = args.data.split(':')
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))

        data_path = paths[0]
        if args.langs is None:
            if args.sampled_data:
                languages = list(cls.get_languages(cls, paths[0]))
            else:
                languages = sorted([
                    name for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ])
        else:
            languages = args.langs.split(',')
        
        dictionary.add_symbol('<mask>')
        if args.add_lang_token:
            if args.common_eos is not None:
                dictionary.add_symbol('[{}]'.format(args.common_eos))
            for lang in languages:
                dictionary.add_symbol('[{}]'.format(lang))
            if args.add_placeholder > 0:
                for i in range(args.add_placeholder):
                    dictionary.add_symbol('[placeholder{}]'.format(i))
            

        logger.info("| dictionary: {} types".format(len(dictionary)))

        if not hasattr(args, 'shuffle_instance'):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.index('<mask>')
        self.langs = args.langs
        self.args = args
        self.path_cache = {}

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def get_languages(self, data_folder):
        files = [path for path in os.listdir(data_folder)]
        lgs = set([x.split('.')[-2] for x in files])
        return lgs

    def get_dataset_path(self, split, data_folder, epoch, lgs=None, is_pair=False):
        if data_folder in self.path_cache:
            files = self.path_cache[data_folder]
        else:
            files = [path for path in os.listdir(data_folder)]
            # remove this to speed up
            # if os.path.isfile(os.path.join(data_folder, path))
            self.path_cache[data_folder] = files

        files = [path for path in files if(split in path) and (".bin" in path)]  

        if lgs is None:
            lgs = set([x.split('.')[-2] for x in files])

        paths = {} 
        for lg_index, lg in enumerate(lgs):
            if is_pair:
                pair = lg.split('-')
                split_count = len([path for path in files if ".{0}.{1}.bin".format(lg, pair[0]) in path])
            else:
                split_count = len([path for path in files if ".{0}.bin".format(lg) in path])
            big_step = epoch // split_count
            small_step = epoch % split_count
            with data_utils.numpy_seed((self.args.seed + big_step) * 100 + lg_index):
                shuffle = np.random.permutation(split_count)
                index = shuffle[small_step]
                path = os.path.join(data_folder, "{0}.{1}.{2}".format(split, index, lg))
                paths[lg] = path
        return paths

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        sampled = self.args.sampled_data
        languages = None if self.langs is None else self.langs.split(',')

        if sampled:
            languages = self.langs if self.langs is None else self.langs.split(',')
            all_lg_path = self.get_dataset_path(split, data_path, epoch, languages)
            if languages is None:
                languages = list(all_lg_path.keys())
        else:
            all_lg_path = None
            if languages is None:
                languages = sorted([
                    name for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ])
            else:
                for name in languages:
                    assert os.path.exists(os.path.join(data_path, name)), "all the languages must exist"

        logger.info("| Training on {0} languages: {1}".format(len(languages), languages))
        logger.info("| Language to id mapping: ", {
                lang: ids for ids, lang in enumerate(languages)
            }
        )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(',')
        lang_datasets = []

        for language in languages:
            split_path = os.path.join(data_path, language, split) if all_lg_path is None else all_lg_path[language]
            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            #end_token = self.source_dictionary.index('[{}]'.format(language)) \
            #    if self.args.add_lang_token else self.source_dictionary.eos()
           
            lg_tag = language if self.args.common_eos is None else self.args.common_eos 
            end_token = self.source_dictionary.index('[{}]'.format(lg_tag)) \
                        if self.args.add_lang_token else self.source_dictionary.eos()

            bos_idx = None
            if self.args.add_lang_token:
                bos_idx = self.source_dictionary.index('[{}]'.format(language))
            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 2,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=end_token,
                break_mode=self.args.sample_break_mode,
            )
            logger.info('| loaded {} blocks from: {}'.format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            if not self.args.no_prepend_sent_bos:
                dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

            if self.args.add_lang_token:
                dataset = PrependTokenDataset(dataset, bos_idx)

            dataset = AppendTokenDataset(dataset, end_token)
            lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None

            lang_dataset = XDAEDenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=None if not self.args.add_lang_token else self.source_dictionary.index('[{}]'.format(lg_tag)),
                bos=bos_idx,
                no_prepend_bos=self.args.no_prepend_sent_bos
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            '| loaded total {} blocks for all languages'.format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            if not self.args.sampled_data:
                #For train subset, additionally up or down sample languages.
                sample_probs = self._get_sample_prob(dataset_lengths)
                logger.info("| Sample probability by language: ", {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                )
                size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
                logger.info("| Up/Down Sampling ratio by language: ", {
                        lang: "{0:.2f}".format(size_ratio[id])
                        for id, lang in enumerate(languages)
                    }
                )

                resampled_lang_datasets = [
                    ResamplingDataset(
                        lang_datasets[i],
                        size_ratio=size_ratio[i],
                        seed=self.args.seed,
                        epoch=epoch,
                        replace=size_ratio[i] >= 1.0,
                    )
                    for i, d in enumerate(lang_datasets)
                ]
                dataset = ConcatDataset(
                    resampled_lang_datasets,
                    )
            else:
                dataset = ConcatDataset(
                    lang_datasets,
                )
        else:
            dataset = ConcatDataset(lang_datasets)

            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + '_' + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

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
