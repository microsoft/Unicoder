# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


task=$1
gid=$2
lg=$3
model=$4
SPE=$5
FD=$6
SPLIT=$7
DATA_PATH=$8
beam_size=$9
CODE_ROOT=${10} 
DATA="${DATA_PATH}/${lg}"
langs=af,als,am,an,ang,ar,arz,ast,az,bar,be,bg,bn,br,bs,ca,ceb,ckb,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gan,gl,gu,he,hi,hr,hu,hy,ia,id,is,it,ja,jv,ka,kk,kn,ko,ku,la,lb,lt,lv,mk,ml,mn,mr,ms,my,nds,ne,nl,nn,no,oc,pl,pt,ro,ru,scn,sco,sh,si,simple,sk,sl,sq,sr,sv,sw,ta,te,th,tl,tr,tt,uk,ur,uz,vi,war,wuu,yi,zh,zh_classical,zh_min_nan,zh_yue


CUDA_VISIBLE_DEVICES=$gid python $CODE_ROOT/generate.py $DATA  --path $model  --task $task --gen-subset $SPLIT -t $lg -s $lg --placeholder 200 --common_eos EOS --bpe 'sentencepiece' --sentencepiece-vocab $SPE --sacrebleu  --remove-bpe 'sentencepiece' --max-sentences 16 --langs $langs --beam $beam_size --no-progress-bar > ${FD}/${lg}_src-tgt

cat ${FD}/${lg}_src-tgt | grep -P "^H" |sort -V |cut -f 3- | sed "s/\[EOS\]//g" > ${FD}/${lg}_tgt.$SPLIT.hyp
