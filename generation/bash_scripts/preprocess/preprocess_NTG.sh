# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


CODE_ROOT=$1     # path to code root
MODEL_DIR=$2     # path/to/saved_model_dir
DATA=$3          # path/to/XGLUE/NTG

SPE_MODEL=$MODEL_DIR/sentencepiece.bpe.model
DICT=$MODEL_DIR/dict.txt

DATA_SPM=$DATA/spm
DATA_BIN=$DATA/bin
DATA_REF=$DATA/ref

mkdir -p $DATA_SPM
mkdir -p $DATA_REF

# Save references

for lg in en es fr de ru; do
    cp ${DATA}/xglue.ntg.$lg.tgt.dev ${DATA_REF}/$lg.tgt.valid 
done


# Tokenize

for lg in en; do
    for split in train dev; do
        for pair in tgt src; do
            echo $lg.$pair.$split
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/xglue.ntg.$lg.$pair.$split --outputs ${DATA}/spm/$lg.$split.spm.$pair
        done
    done
done

for lg in es fr de ru; do
    for split in dev; do
        for pair in tgt src; do
            echo $lg.$pair.$split
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/xglue.ntg.$lg.$pair.$split --outputs ${DATA}/spm/$lg.$split.spm.$pair
        done
    done
done


for lg in en es fr de ru; do
    for split in test; do
        for pair in src; do
            echo $lg.$pair.$split
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/xglue.ntg.$lg.$pair.$split --outputs ${DATA}/spm/$lg.$split.spm.$pair
        done
    done
done


# Truncate source to 512

python $CODE_ROOT/bash_scripts/preprocess/truncate_src.py --path $DATA_SPM --max_len 512


# Binarize

for lg in en es fr de ru; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
	python $CODE_ROOT/preprocess.py \
	--source-lang src \
    --target-lang tgt \
    --only-source \
	--testpref $DATA_SPM/$lg.test.spm \
	--destdir $DATA_BIN/$lg \
	--thresholdtgt 0 \
	--thresholdsrc 0 \
	--srcdict ${DICT} \
	--workers 120
done


for lg in en; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
    python $CODE_ROOT/preprocess.py \
    --source-lang src \
    --target-lang tgt \
    --trainpref $DATA_SPM/$lg.train.spm \
    --validpref $DATA_SPM/$lg.dev.spm \
    --destdir $DATA_BIN/$lg \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 120
done

for lg in es fr de ru; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
    python $CODE_ROOT/preprocess.py \
    --source-lang src \
    --target-lang tgt \
    --validpref $DATA_SPM/$lg.dev.spm \
    --destdir $DATA_BIN/$lg \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 120
done

echo "Done!"
