#!/bin/bash

if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside an Apptainer container."
else
    PYTHON="$HOME/Installs/conda/envs/clipcount/bin/python"
    # PYTHON=/opt/conda/bin/python
    echo "Not running inside an Apptainer container."
fi

CUDA_VISIBLE_DEVICES=0 $PYTHON run.py \
    --mode test \
    --exp_name g2iete_1_test \
    --task_name g2iete_1_test \
    --batch_size 1 \
    --dataset_type FSC_gdino \
    --ckpt lightning_logs/g2iete_1/version_0/checkpoints/epoch=181-val_mae=13.34.ckpt\
    --g_logits g2iete
