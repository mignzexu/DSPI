#!/bin/bash

if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/opt/conda/bin/python
    echo "Running inside an Apptainer container."
else
    PYTHON="$HOME/Installs/conda/envs/clipcount/bin/python"
    echo "Not running inside an Apptainer container."
fi

# ada_ql_values=(1 2 4 8)
# ada_ql_values=(16 32 64 128)
ada_ql_values=(256)
num=$1

for i in "${!ada_ql_values[@]}"; do
  cuda_device=$((i % 4))
  ada_ql=${ada_ql_values[$i]}
  CUDA_VISIBLE_DEVICES=$cuda_device $PYTHON run.py \
    --mode train \
    --exp_name g2i_ada_ql_${ada_ql}_v${num} \
    --task_name g2i_ada_ql_${ada_ql}_v${num} \
    --epochs 200 \
    --dataset_type FSC_gdino \
    --g_logits g2i \
    --ada_ql $ada_ql &
done

wait