# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


lg=$1          # supervised lanugage for finetuning [en]
NGPU=$2        # num of GPUs to use
CODE_ROOT=$3   # path/to/code_root
MODEL_DIR=$4   # path/to/model_dir
OUTPUT_DIR=$5  # output dir to save checkpoints, decodings, etc 
DATA_ROOT=$6   # path/to/XGLUE/QG  


PRETRAIN=$MODEL_DIR/checkpoint.pt
SPE=$MODEL_DIR/sentencepiece.bpe.model

DATA_BIN=$DATA_ROOT/bin
DATA_REF=$DATA_ROOT/ref

langs=af,als,am,an,ang,ar,arz,ast,az,bar,be,bg,bn,br,bs,ca,ceb,ckb,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gan,gl,gu,he,hi,hr,hu,hy,ia,id,is,it,ja,jv,ka,kk,kn,ko,ku,la,lb,lt,lv,mk,ml,mn,mr,ms,my,nds,ne,nl,nn,no,oc,pl,pt,ro,ru,scn,sco,sh,si,simple,sk,sl,sq,sr,sv,sw,ta,te,th,tl,tr,tt,uk,ur,uz,vi,war,wuu,yi,zh,zh_classical,zh_min_nan,zh_yue

lr=1e-5

TBS=1024
max_sents=16
update_freq=$(($TBS/$max_sents/$NGPU))

warmup=2000
mepoch=50

task=generation_from_pretrained_bart
EXP="FINETUNE_QG_${lg}"

SAVE=${OUTPUT_DIR}/$EXP

mkdir -p $SAVE

SUFFIX=""
if [ ! -f $SAVE/checkpoint_last.pt ]; then
   echo "copy pretrained model to last"
   cp $PRETRAIN $SAVE/checkpoint_last.pt
fi

if [ ! -f $SAVE/checkpoint1.pt ]; then
   SUFFIX="$SUFFIX --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer"
fi

python $CODE_ROOT/train.py ${DATA_BIN}/${lg}  \
           --save-dir $SAVE \
           --arch mbart_base \
           --encoder-layers 12 \
           --decoder-layers 12 \
           --max-source-positions 512 \
           --max-target-positions 512 \
           --disable-validation \
           --task $task \
           --source-lang $lg \
           --target-lang $lg \
           --criterion label_smoothed_cross_entropy \
           --label-smoothing 0.2  \
           --common_eos EOS \
           --placeholder 200 \
           --dataset-impl mmap \
           --optimizer adam \
           --adam-eps 1e-06 \
           --adam-betas '(0.9, 0.98)' \
           --lr-scheduler inverse_sqrt \
           --lr $lr --min-lr -1 \
           --warmup-updates $warmup \
           --dropout 0.1 \
           --attention-dropout 0.1  \
           --weight-decay 0.01 \
           --max-sentences $max_sents \
           --update-freq $update_freq \
           --save-interval 1 \
           --save-interval-updates 100000 \
           --keep-interval-updates 1 \
           --max-epoch $mepoch \
           --seed 1023 \
           --log-format simple --log-interval 2 \
           --langs $langs \
           --layernorm-embedding  --ddp-backend no_c10d --fp16 \
           $SUFFIX


python $CODE_ROOT/evaluation/decode_all.py --task $task --beam 5 --start 1 --ngpu ${NGPU} --epoch ${mepoch} \
           --split valid --exp $EXP --data_path $DATA_BIN --spe $SPE --save_dir $OUTPUT_DIR --dataset QG \
           --code_root $CODE_ROOT

python $CODE_ROOT/evaluation/eval_exp.py --task $task --test_beam 10 --ngpu ${NGPU} --epoch ${mepoch} \
           --exp $EXP --data_path $DATA_BIN --ref_folder $DATA_REF \
           --lgs en-fr-es-de-it-pt --supervised_lg $lg --spe $SPE --save_dir $OUTPUT_DIR --dataset QG \
           --code_root $CODE_ROOT
