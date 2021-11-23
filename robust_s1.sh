

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
python main.py --nStart 1 --model ResNet18  --nEnd 40 --nQuery 5 --dataset cifar10 --save_file result_s1.csv --save_path save/coreGCN_rob/ --strategy coreGCN --rand_idx 1
