# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TASK=$1
NGPU=$2
EXP=$3
EPOCH=$4
DATASET=$5
SPLIT=$6
SPE=$7
DATA_PATH=$8
BEAM=$9
ST=${10}
SAVE_DIR=${11}
CODE_ROOT=${12}

gid=0

for ((e=$EPOCH; e>=$ST; e--))
do
    model="${SAVE_DIR}/${EXP}/checkpoint${e}.pt"
    path="${SAVE_DIR}/decodes/${EXP}/checkpoint${e}/$SPLIT"
    mkdir -p $path
    if [ $DATASET == "NTG" ]; then
        for lg in en fr es de ru; do
            echo $model $lg $SPLIT
            bash ${CODE_ROOT}/evaluation/generate_single.sh $TASK $gid $lg $model $SPE $path $SPLIT $DATA_PATH $BEAM $CODE_ROOT & 
            gid=$(($gid+1)) 
            if [ $(($gid%$NGPU)) = 0 ]; then
                wait
                gid=0
            fi
        done
    fi
    if [ $DATASET == "QG" ]; then
        for lg in en fr de es it pt; do
            echo $model $lg $SPLIT
            bash ${CODE_ROOT}/evaluation/generate_single.sh $TASK $gid $lg $model $SPE $path $SPLIT $DATA_PATH $BEAM $CODE_ROOT & 
            gid=$(($gid+1)) 
            if [ $(($gid%$NGPU)) = 0 ]; then
                wait
                gid=0
            fi
        done
    fi
done

wait
echo "decoding done!"
