#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_vmem=7.5G
#$ -l h_rt=240:00:0
#$ -l gpu=1
#$ -m bea
#$ -M qilei.li@hotmail.com
#$ -l gpu_type=ampere
#$ -N clipcount

if [[ "$HOSTNAME" == *sbg* || "$HOSTNAME" == *rdg* ]]; then
    TMP_CODE_DIR="/data/DERI-Gong/ql001/Experiments/ClipCount"
else
    TMP_CODE_DIR="/home/wenzhe/Experiments/heu/ClipCount"
fi

DATASET="FSC_gdino"
TASK="debug"
GD="None"
MA=""

while [[ $# -gt 0 ]]; do
    case $1 in
    -p | --tmp)
        TMP_CODE_DIR=$2
        shift 2
        ;;
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

# apptainer exec --nv $HOME/Installs/containers/clipcount_gdino_v2.sif bash -c "cd $TMP_CODE_DIR && bash scripts/train.sh -d $DATASET -t $TASK -g $GD $([[ $MA ]] && echo "--ma $MA" || echo "")"
# apptainer exec --nv $HOME/Installs/containers/clipcount_gdino_v2.sif bash -c "cd $TMP_CODE_DIR && bash scripts/online_test.sh -d JHU"
# cd $TMP_CODE_DIR && bash scripts/train.sh -d $DATASET -t $TASK -g $GD $([[ $MA ]] && echo "--ma $MA" || echo "")

apptainer exec --nv --bind /scratch:/scratch --bind /shares:/shares /shares/containers/clipcount_v2.sif bash -c "cd $HOME/Experiments/ClipCount && bash scripts/test.sh"
