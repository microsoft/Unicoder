# Unicoder
This repo provides the code for reproducing the experiment in [XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation](https://arxiv.org/abs/2004.01401).

## Install and Dependency

This repo is based on [Transformers](https://github.com/huggingface/transformers). It's tested on Python 3.6+, PyTorch 1.4.0.

Installing this repo will replaced the original Transformers you installed. So we recommend you install this repo in a separate [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

You could install this repo with pip by:

```bash
git clone https://github.com/microsoft/Unicoder
cd Unicoder/understanding
pip install . 
```

You could speed up all experiments by installing [apex](https://github.com/NVIDIA/apex) and setting --fp16.

## Pre-trained model
The pre-trained model used in paper is at [here](https://1drv.ms/u/s!Amt8n9AJEyxckUl6e1nXDQmppZFc?e=SE8laa).
## Fine-tuning experiments

### XGLUE dataset
You can download XGLUE dataset from [XGLUE homepage](https://microsoft.github.io/XGLUE/).

### Fine-tuning
We used a single V100 with 32GB memory to run all the experiments. If you are using GPU with 16GB memory or less, you could decrease the per_gpu_train_batch_size and increase gradient_accumulation_steps. 

In our experiments, we used FP16 for speed up.
It's known that using FP32 or K80/M60 GPU may lead to slightly different performance.

You could run the [xglue.sh](unicoder/xglue.sh) to reproduce all the experiment.

xglue.sh has three parameters. 
```bash
DATA_DIR: downloaded from XGLUE github
MODEL_DIR: downloaded from section "pre-trained model"
OUTPUT_DIR: any folder
```

#### XNLI:
```bash
python examples/run_xglue.py --model_type xlmr \
--model_name_or_path $MODEL_DIR \
--language ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
--train_language en \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR/XNLI \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--max_seq_length 256 \
--output_dir $OUTPUT_DIR/XNLI \
--task_name xnli \
--save_steps -1 \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
--logging_steps -1 \
--logging_steps_in_sample -1 \
--logging_each_epoch \
--gpu_id 0
```

#### QADSM:
```bash
python examples/run_xglue.py --model_type xlmr \
--model_name_or_path $MODEL_DIR \
--language de,en,fr \
--train_language en \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR/QADSM \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--max_seq_length 256 \
--output_dir $OUTPUT_DIR/QADSM \
--task_name ads \
--save_steps -1 \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
--logging_steps -1 \
--logging_steps_in_sample -1 \
--logging_each_epoch \
--gpu_id 0
```

#### QAM:
```bash
python examples/run_xglue.py --model_type xlmr \
--model_name_or_path $MODEL_DIR \
--language de,en,fr \
--train_language en \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR/QAM \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--max_seq_length 256 \
--output_dir $OUTPUT_DIR/QAM \
--task_name qam \
--save_steps -1 \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
--logging_steps -1 \
--logging_steps_in_sample -1 \
--logging_each_epoch \
--gpu_id 0
```

#### PAWSX:
```bash
python examples/run_xglue.py --model_type xlmr \
--model_name_or_path $MODEL_DIR \
--language de,en,es,fr \
--train_language en \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR/PAWSX \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--max_seq_length 256 \
--output_dir $OUTPUT_DIR/PAWSX \
--task_name pawsx \
--save_steps -1 \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
--logging_steps -1 \
--logging_steps_in_sample -1 \
--logging_each_epoch \
--gpu_id 0
```

#### NC:
```bash
python examples/run_xglue.py --model_type xlmr \
--model_name_or_path $MODEL_DIR \
--language de,en,es,fr,ru \
--train_language en \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR/NC \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--max_seq_length 256 \
--output_dir $OUTPUT_DIR/NC \
--task_name news \
--save_steps -1 \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
--logging_steps -1 \
--logging_steps_in_sample -1 \
--logging_each_epoch \
--gpu_id 0
```

#### WPR:
```bash
python examples/run_xglue.py --model_type xlmr \
--model_name_or_path $MODEL_DIR \
--language de,en,es,fr,it,pt,zh \
--train_language en \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR/WPR \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--max_seq_length 256 \
--output_dir $OUTPUT_DIR/WPR \
--task_name rel \
--save_steps -1 \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
--logging_steps -1 \
--logging_steps_in_sample -1 \
--logging_each_epoch \
--gpu_id 0
```

#### MLQA:
```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_xmrc.py --model_type xlmr  \
--model_name_or_path $MODEL_DIR \
--do_train \
--do_eval \
--do_lower_case \
--language en,es,de,ar,hi,vi,zh \
--train_language en \
--data_dir $DATA_DIR/MLQA \
--per_gpu_train_batch_size 12 \
--per_gpu_eval_batch_size 128 \
--learning_rate 3e-5 \
--num_train_epochs 2.0 \
--save_steps 0 \
--logging_each_epoch \
--max_seq_length 384 \
--doc_stride 128  \
--output_dir $OUTPUT_DIR/MLQA \
--overwrite_output_dir \
--overwrite_cache \
--evaluate_during_training \
```

#### NER:
```bash
python examples/ner/run_ner.py --data_dir $DATA_DIR/NER \
--model_type xlmroberta \
--labels $DATA_DIR/NER/labels.txt \
--model_name_or_path $MODEL_DIR \
--gpu_id 0 \
--output_dir $OUTPUT_DIR/NER \
--language de,en,es,nl \
--train_language en \
--max_seq_length 256 \
--num_train_epochs 20 \
--per_gpu_train_batch_size 32 \
--save_steps 1500 \
--seed 42 \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir \
--overwrite_cache \
--logging_each_epoch \
--evaluate_during_training \
--learning_rate 5e-6 \
--task_name ner
```

#### POS:
```bash
python examples/ner/run_ner.py --data_dir $DATA_DIR/POS \
--model_type xlmroberta \
--labels $DATA_DIR/POS/labels \
--model_name_or_path $MODEL_DIR \
--gpu_id 0 \
--output_dir $OUTPUT_DIR/POS \
--language en,ar,bg,de,el,es,fr,hi,it,nl,pl,pt,ru,th,tr,ur,vi,zh \
--train_language en \
--max_seq_length 128 \
--num_train_epochs 20 \
--per_gpu_train_batch_size 32 \
--save_steps 1500 \
--seed 42 \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir \
--overwrite_cache \
--logging_each_epoch \
--evaluate_during_training \
--learning_rate 2e-5 \
--task_name pos
```


The languages in fine-tuning and testing doesn't need to be same. You could only fine-tune on English and test on other languages. Based on our experiments, translate English training data to other languages and fine-tune on it could improve the performance.



## Notes and Acknowledgments
This code base is built on top of [Transformers](https://github.com/huggingface/transformers).
### Revised Files
examples/run_xglue.py

examples/run_xglue_ft.py

examples/run_xmrc.py

examples/ner/run_ner.py

src/transformers/data/processors/xglue.py

src/transformers/data/metrics/__init__.py

### Aadded Files
Unicoder/xglue.sh
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
