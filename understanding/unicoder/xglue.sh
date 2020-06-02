# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR=$3

mkdir -p $OUTPUT_DIR


#xnli:
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

#qadsm:
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

#qam:
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

#pawsx:
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
--logging_each_epoch --gpu_id 0

#nc:
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

#wpr:
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


#mlqa:

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


#ner
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

# pos
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


