#!/bin/bash

CMD="bash"

while [[ $# -gt 0 ]]; do
    case $1 in
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

for count in 1 2 3; do
    bash scripts/sync.sh -c $CMD --task inf/abl/baseline/noma_nogd_FSC_${count}
done

for ma in 1 2 4 8 16 32 64 128; do
    bash scripts/sync.sh -c $CMD --ma $ma --task inf/abl/ma/ma_${ma}_nogd_FSC
done

for count in 1 2 3; do
    bash scripts/sync.sh -c $CMD --gd g2i --task inf/abl/gd/gd_FSC_${count}
done

for ma in 1 2 4 8 16 32 64 128; do
    bash scripts/sync.sh -c $CMD --ma $ma --gd g2i --task inf/abl/ma/ma_${ma}_gd_g2i_FSC
done


# bash scripts/sync.sh -c bash --task inf/abl/debug
