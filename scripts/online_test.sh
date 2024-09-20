#!/bin/bash

if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside an Apptainer container."
else
    PYTHON="$HOME/Installs/conda/envs/clipcount/bin/python"
    echo "Not running inside an Apptainer container."
fi

if [[ "$HOSTNAME" == *sbg* || "$HOSTNAME" == *rdg* ]]; then
    BASE_PATH="/data/DERI-Gong/ql001/Experiments/ClipCount"
    ORI_PATH="/data/DERI-Gong/ql001/Experiments/ClipCount/data"
else
    BASE_PATH="/home/wenzhe/Experiments/heu/ClipCount"
    ORI_PATH="/shares/crowd_counting/CLIP-Count_data"
    TMPDIR="/scratch"
fi

DATASET="FSC_gdino"
TASK="debug"
GD=g2i
MA=16
SUBSET=A

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
    -* | --*)
        echo "Unknown option $1"
        exit 1
        ;;
    esac
done

if [[ $DATASET == "FSC_gdino" ]]; then
    DATANAME="FSC"
else
    DATANAME=$DATASET
fi

# if [ ! -d "$TMPDIR/CLIP-Count_data/$DATANAME" ]; then
#     current_dir=$(pwd)
#     rsync -arvP $ORI_PATH/$DATANAME.zip $TMPDIR/CLIP-Count_data/
#     cd $TMPDIR/CLIP-Count_data && unzip $DATANAME.zip
#     rm $DATANAME.zip
#     cd "$current_dir"
# fi

BESTMODEL=$BASE_PATH/lightning_logs/inf/abl/ma/ma_16_gd_g2i_FSC/version_0/checkpoints/epoch=113-val_mae=13.10.ckpt
# BESTMODEL=/scratch/CLIP-Count_data/lightning_logs/inf/abl/ma/ma_16_gd_g2i_FSC/version_0/checkpoints/epoch=113-val_mae=13.10.ckpt
TASK=$(echo "$BESTMODEL" | sed 's|.*lightning_logs/\(.*\)/version.*|\1|')

CUDA_VISIBLE_DEVICES=0 $PYTHON run.py \
    --data_path data \
    --mode test \
    --exp_name $TASK \
    --batch_size 1 \
    --dataset_type $DATASET \
    $([[ $DATASET == ShanghaiTech ]] && echo "--sh_dataset $SUBSET" || echo "") \
    $([[ $DATASET == UCF50 ]] && echo "--subset $SUBSET" || echo "") \
    --ckpt $BESTMODEL \
    --g_logits $GD \
    $([[ $MA ]] && echo "--ma --ada_ql $MA" || echo "") \
    --online
# --log_test_img True \
# --log_dir out/$TASK/$([[ $DATASET == ShanghaiTech ]] && echo "${DATASET}_$SUBSET" || echo "$DATASET") \
# --attn_map

# bash scripts/online_test.sh -d ShanghaiTech -s A
# bash scripts/online_test.sh -d ShanghaiTech -s B
# bash scripts/online_test.sh -d CARPK
# bash scripts/online_test.sh -d FSC_gdino
# bash scripts/online_test.sh -d UCF50 -s 0
