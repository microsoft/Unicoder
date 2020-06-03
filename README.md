# Unicoder
This repo provides the code for reproducing the experiments in [XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation](https://arxiv.org/abs/2004.01401) (`[leaderboard](https://microsoft.github.io/XGLUE/, "https://microsoft.github.io/XGLUE/")`).

We provide three models, Unicoder for understanding tasks, Unicoder for generation tasks (pre-trained with xDAE) and Unicoder for generation tasks (pre-trained with xFNP).

## Unicoder for understanding tasks
We share a 12-layers model which is pre-trained with 100 languages.

This code can reproduce the experiments on 9 understanding XGLUE tasks: NER,
POS Tagging (POS),
News Classification (NC),
MLQA,
XNLI,
PAWS-X,
Query-Ad Matching (QADSM),
Web Page Ranking (WPR),
QA Matching (QAM).

For more details, you can go to [understanding README](./understanding/README.md).

## Unicoder for generation tasks (pre-trained with xDAE)
We share a 12-layer encoder and 12-layer decoder model which is pre-trained with 100 languages.

The code can reproduce the experiments on 2 generation XGLUE tasks: News Title Generation(NTG) and Question Generation (QG).

For more details, you can go to [generation README](./generation/README.md).

## Unicoder for generation tasks (pre-trained with xFNP)
We share a 12-layer encoder and 12-layer decoder model which is pre-trained with 100 languages.

The code can reproduce the experiments on 2 generation XGLUE tasks: News Title Generation(NTG) and Question Generation (QG).

For more details, you can go to [ProphetNet](https://github.com/microsoft/ProphetNet/tree/master/xProphetNet).

## How to cite
If you extend or use this work, please cite [our paper](https://arxiv.org/abs/2004.01401).
```
@inproceedings{huang2019unicoder,
  title={Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks},
  author={Huang, Haoyang and Liang, Yaobo and Duan, Nan and Gong, Ming and Shou, Linjun and Jiang, Daxin and Zhou, Ming},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2485--2494},
  year={2019}
}
@article{Liang2020XGLUEAN,
  title={XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation},
  author={Yaobo Liang and Nan Duan and Yeyun Gong and Ning Wu and Fenfei Guo and Weizhen Qi and Ming Gong and Linjun Shou and Daxin Jiang and Guihong Cao and Xiaodong Fan and Ruofei Zhang and Rahul Agrawal and Edward Cui and Sining Wei and Taroon Bharti and Ying Qiao and Jiun-Hung Chen and Winnie Wu and Shuguang Liu and Fan Yang and Daniel Campos and Rangan Majumder and Ming Zhou},
  journal={arXiv},
  year={2020},
  volume={abs/2004.01401}
}
```

## More Models in the Unicoder Family (`coming soon`)
[Unicoder-VL (image)](https://arxiv.org/abs/1908.06066 "https://arxiv.org/abs/2003.01473"): a monolingual (English) pre-trained model for image-language understanding tasks.  
[Unicoder-VL (video)](https://arxiv.org/abs/2002.06353 "https://arxiv.org/abs/2003.01473"): a monolingual (English) pre-trained model for video-language understanding and generation tasks.  
[XGPT (image)](https://arxiv.org/abs/2003.01473 "https://arxiv.org/abs/2003.01473"): a monolingual (English) pre-trained model for image captioning.  
[M^3P (image)](): a multilingual (100 languages) pre-trained model for image-language understnading and generation tasks.  
[CodeBERT](https://arxiv.org/abs/2002.08155 "https://arxiv.org/abs/2002.08155"): a multi-Plingual pre-trained model for NL(natural language)-PL(programming language) tasks. 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
