#!/bin/bash

if [ ! -d "./save/" ]; then
    mkdir ./save/
fi

DIRECTORY=./save/ClusterMarginSampling_rob/
if [ ! -d "$DIRECTORY" ]; then
    mkdir ${DIRECTORY}
fi
DIRECTORY=./save/BALDDropout_rob/
if [ ! -d "$DIRECTORY" ]; then
    mkdir ${DIRECTORY}
fi
DIRECTORY=./save/coreGCN_rob/
if [ ! -d "$DIRECTORY" ]; then
    mkdir ${DIRECTORY}
fi
DIRECTORY=./save/LeastConfidence_rob/
if [ ! -d "$DIRECTORY" ]; then
    mkdir ${DIRECTORY}
fi

export CUBLAS_WORKSPACE_CONFIG=:16:8
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/coreGCN_rob/ --strategy coreGCN --rand_idx 1
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/coreGCN_rob/ --strategy coreGCN --rand_idx 2
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/coreGCN_rob/ --strategy coreGCN --rand_idx 3
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/coreGCN_rob/ --strategy coreGCN --rand_idx 4
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/coreGCN_rob/ --strategy coreGCN --rand_idx 5
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/LeastConfidence_rob/ --strategy LeastConfidence --rand_idx 1
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/LeastConfidence_rob/ --strategy LeastConfidence --rand_idx 2
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/LeastConfidence_rob/ --strategy LeastConfidence --rand_idx 3
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/LeastConfidence_rob/ --strategy LeastConfidence --rand_idx 4
python main.py --nStart 2 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s2.csv --save_path save/LeastConfidence_rob/ --strategy LeastConfidence --rand_idx 5