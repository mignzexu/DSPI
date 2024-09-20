#!/bin/bash
if [[ -n "$APPTAINER_CONTAINER" || -n "$DOCKER_CONTAINER" ]]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside an Apptainer container."
else
    PYTHON="$HOME/Installs/conda/envs/clipcount/bin/python"
    echo "Not running inside an Apptainer container."
fi

# $PYTHON run.py \
#     --mode app \
#     --exp_name debug \
#     --data_path $TMPDIR/CLIP-Count_data \
#     --batch_size 1 \
#     --dataset_type FSC_gdino \
#     --g_logits g2iete \
#     --ma \
#     --ada_ql 8 \
#     --ckpt ./lightning_logs/best/g2iete_2_macb2_8/version_0/checkpoints/epoch=195-val_mae=12.39.ckpt

$PYTHON run.py \
    --mode app \
    --exp_name debug \
    --data_path $TMPDIR/CLIP-Count_data \
    --batch_size 1 \
    --dataset_type FSC \
    --ckpt ./pretrained/clipcount_pretrained.ckpt
