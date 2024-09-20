#!/bin/bash

if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside an Apptainer container."
elif [ -n "$DOCKER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside a Docker container."
else
    PYTHON="$HOME/Installs/conda/envs/clipcount/bin/python"
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

DATASET="FSC_gdino"
TASK="debug"
GD="g2i"
MA="16"

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
    -* | --*)
        echo "Unknown option $1"
        exit 1
        ;;
    esac
done

if [[ $DATASET == "FSC_gdino" ]]; then
    DATANAME="FSC"
fi

if [ ! -d "$TMPDIR/CLIP-Count_data/$DATANAME" ]; then
    current_dir=$(pwd)
    rsync -arvP $ORI_PATH/$DATANAME.zip $TMPDIR/CLIP-Count_data/
    cd $TMPDIR/CLIP-Count_data && unzip $DATANAME.zip
    rm $DATANAME.zip
    cd "$current_dir"
fi

CUDA_VISIBLE_DEVICES=0 $PYTHON run.py \
    --data_path $TMPDIR/CLIP-Count_data \
    --output_dir $BASE_PATH/out \
    --results $BASE_PATH/lightning_logs \
    --exp_name $TASK \
    --g_logits $GD \
    --dataset_type $DATASET \
    --mode train \
    --batch_size 32 \
    --epochs 200 \
    $([[ $MA ]] && echo "--ma --ada_ql $MA" || echo "")
