#!/bin/bash

if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside an Apptainer container."
elif [ -n "$DOCKER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside a Docker container."
else
    PYTHON=/opt/conda/bin/python
    echo "Not running inside an Apptainer container."
fi

if [[ "$HOSTNAME" == *sbg* || "$HOSTNAME" == *rdg* ]]; then
    BASE_PATH="/data/DERI-Gong/ql001/Experiments/ClipCount"
    ORI_PATH="/data/DERI-Gong/ql001/Experiments/ClipCount/data"
else
    BASE_PATH="$HOME/Experiments/ClipCount"
    ORI_PATH="/shares/crowd_counting/CLIP-Count_data"
    TMPDIR="/scratch"
fi

DATASET="NWPU"
TASK="debug"
GD="g2i"
MA=16
BASELINE=""
VERBOSE=""

SUBSET=val

while [[ $# -gt 0 ]]; do
    case $1 in
    -d | --dataset)
        DATASET=$2
        shift 2
        ;;
    -t | --task)
        TASK=$2
        shift 2
        ;;
    -g | --gd)
        GD=$2
        shift 2
        ;;
    -m | --ma)
        MA=$2
        shift 2
        ;;
    -s | --subset)
        SUBSET=$2
        shift 2
        ;;
    -b | --baseline)
        BASELINE="True"
        shift 1
        ;;
    -v | --verbose)
        VERBOSE="True"
        shift 1
        ;;
    -* | --*)
        echo "Unknown option $1"
        exit 1
        ;;
    esac
done

# if [ ! -d "$TMPDIR/CLIP-Count_data/FSC" ]; then
#     current_dir=$(pwd)
#     rsync -arvP $ORI_PATH/FSC.zip $TMPDIR/CLIP-Count_data/
#     cd $TMPDIR/CLIP-Count_data && unzip FSC.zip
#     rm FSC.zip
#     cd "$current_dir"
# fi

# if [[ $DATASET != "FSC_gdino" ]] && [[ ! -d "$TMPDIR/CLIP-Count_data/$DATASET" ]]; then
#     current_dir=$(pwd)
#     rsync -arvP $ORI_PATH/$DATASET.zip $TMPDIR/CLIP-Count_data/
#     cd $TMPDIR/CLIP-Count_data && unzip $DATASET.zip
#     rm $DATASET.zip
#     cd "$current_dir"
# fi

if [[ $BASELINE ]]; then
    MA=""
    GD="None"
    # BESTMODEL=$BASE_PATH/release/baseline_epoch=192-val_mae=15.29.ckpt
    BESTMODEL="release/baseline_epoch=192-val_mae=15.29.ckpt"
else
    # BESTMODEL=$BASE_PATH/release/dspi_epoch=113-val_mae=13.10.ckpt
    BESTMODEL="release/dspi_epoch=113-val_mae=13.10.ckpt"
fi

# TASK=$(echo "$BESTMODEL" | sed 's|.*lightning_logs/\(.*\)/version.*|\1|')

TASK=baseline

CUDA_VISIBLE_DEVICES=0 $PYTHON run.py \
    --data_path $TMPDIR/CLIP-Count_data \
    --mode test \
    --exp_name $TASK \
    --batch_size 1 \
    --dataset_type $DATASET \
    $([[ $DATASET == ShanghaiTech ]] && echo "--sh_dataset $SUBSET" || echo "") \
    $([[ $DATASET == UCF50 ]] && echo "--subset $SUBSET" || echo "") \
    $([[ $DATASET == NWPU ]] && echo "--split $SUBSET" || echo "") \
    --ckpt $BESTMODEL \
    --g_logits $GD \
    $([[ $MA ]] && echo "--ma --ada_ql $MA" || echo "") \
    --log_test_img True \
    --log_dir $BASE_PATH/out/$TASK/$([[ $DATASET == ShanghaiTech || $DATASET == UCF50 || $DATASET == NWPU ]] && echo "${DATASET}_$SUBSET" || echo "$DATASET") \
    --attn_map \
    $([[ $VERBOSE ]] && echo "--verbose")

# bash scripts/test.sh -d ShanghaiTech -s A
# bash scripts/test.sh -d ShanghaiTech -s B
# bash scripts/test.sh -d CARPK
# bash scripts/test.sh -d PUCPR
# bash scripts/test.sh -d QNRF
# bash scripts/test.sh -d JHU
# bash scripts/test.sh -d FSC_gdino
# bash scripts/test.sh -d UCF50 -s 1 -b
