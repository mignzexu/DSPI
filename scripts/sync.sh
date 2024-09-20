#!/bin/bash

DATASET="FSC_gdino"
TASK="debug"
GD="None"
MA=""
CMD=bash

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
    -c | --cmd)
        CMD=$2
        shift 2
        ;;
    -* | --*)
        echo "Unknown option $1"
        exit 1
        ;;
    esac
done

CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
REMOTE_CODE="."

TMP_CODE_DIR="$HOME/Cache/Code/ClipCount/${CURRENT_TIME}_$(echo "$TASK" | sed 's|/|-|g')"

# Create the temporary directory
mkdir -p "$TMP_CODE_DIR"

echo "Start code transfer at $(date)"

# Define the excluded patterns
exclusions=(
  --exclude results
  --exclude '*.pth'
  --exclude '*.pyc'
  --exclude '*.o*'
  --exclude '*.sif'
  --exclude '*.zip'
  --exclude .git
  --exclude GroundingDINO/pretrained
  --exclude data
  --exclude out
  --exclude lightning_logs
  --exclude debug 
)

# Sync the code with rsync
rsync -a "$REMOTE_CODE" "$TMP_CODE_DIR" "${exclusions[@]}"
echo "Finish code transfer at $(date)"

cd $TMP_CODE_DIR && echo Working under $PWD
$CMD scripts/sub.sh -p $TMP_CODE_DIR -d $DATASET -t $TASK -g $GD $([[ $MA ]] && echo "--ma $MA" || echo "")
