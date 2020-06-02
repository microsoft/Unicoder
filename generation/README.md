# Unicoder - Cross-lingual Generation

This repo provides the code for reproducing the experiment in [XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation](https://arxiv.org/abs/2004.01401).



## Requirements and Installation

* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

**Installing from source**

```bash
git clone git@github.com:microsoft/Unicoder.git 
cd Unicoder/generation
pip install --editable .
```


## Preprocess data for finetuning 

### Download XGLUE dataset

You can download XGLUE dataset from [XGLUE homepage](https://microsoft.github.io/XGLUE/).

### Preprocess NTG data

```bash
bash ./bash_scripts/preprocess/preprocess_NTG.sh \
         path/to/code_root \
         path/to/model_root_dir \
         path/to/XGLUE/NTG
```

### Preprocess QG data

```bash
bash ./bash_scripts/preprocess/preprocess_QG.sh \
         path/to/code_root \
         path/to/model_root_dir \
         path/to/XGLUE/QG
```


## Generation fine-tuning with XDAE

  
  ### Download pretrained model

  You can download the pretrained XDAE model used in this paper [here](https://1drv.ms/u/s!Amt8n9AJEyxckWbpMyGKPKWDjTG-?e=elsf31).

  ### Finetune with one supervised language and run zero-shot decoding with multilingual data

  #### NTG

```bash
bash ./bash_scripts/finetune/finetune_NTG.sh \
         en[supervised language] \
         8[num of GPUs on your machine] \
         path/to/code_root \
         path/to/model_dir \
         output_dir \
         path/to/XGLUE/NTG

# By default, the code uses all GPUs on your machine, and you should pass the number of GPUs anyway. 
# To use a subset of the GPUs, 
# specify the GPU ids with CUDA_VISIBLE_DEVICES=x,x,..,x and change the number of GPUs accordingly. 
```
  #### QG

```bash
bash ./bash_scripts/finetune/finetune_QG.sh \
         en[supervised language] \
         8[num of GPUs on your machine] \
         path/to/code_root \
         path/to/model_dir \
         output_dir \
         path/to/XGLUE/QG
```


## Notes and Acknowledgments

This code base is built on top of [FAIRSEQ](https://github.com/pytorch/fairseq).

#### Added tasks and datasets for generation
  
  generation/fairseq/tasks/generation_from_pretrained_bart.py

  generation/fairseq/tasks/generation_from_pretrained_xlmr.py

  generation/fairseq/tasks/multilingual_generation_from_bart.py

  generation/fairseq/tasks/multilingual_denoising_xdae.py

  generation/fairseq/data/generation_pair_dataset.py

  generation/fairseq/data/generation_multi_pair_dataset.py

  generation/fairseq/data/xdae_denoising_dataset.py

#### Added scripts

  generation/evaluation
  generation/bash_scripts

## How to cite

If you extend or use this work please cite [our paper](https://arxiv.org/abs/2004.01401).
```
@article{Liang2020XGLUEAN,
  title={XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation},
  author={Yaobo Liang and Nan Duan and Yeyun Gong and Ning Wu and Fenfei Guo and Weizhen Qi and Ming Gong and Linjun Shou and Daxin Jiang and Guihong Cao and Xiaodong Fan and Ruofei Zhang and Rahul Agrawal and Edward Cui and Sining Wei and Taroon Bharti and Ying Qiao and Jiun-Hung Chen and Winnie Wu and Shuguang Liu and Fan Yang and Daniel Campos and Rangan Majumder and Ming Zhou},
  journal={arXiv},
  year={2020},
  volume={abs/2004.01401}
}
```
