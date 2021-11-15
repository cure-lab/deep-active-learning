#!/bin/bash
#SBATCH --job-name=TEST
#SBATCH --mail-user=yuli@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=./save/
#SBATCH --gres=gpu:2 

HOST=$(hostname)
echo "Current host is: $HOST"

DATE=`date +%Y-%m-%d`
echo $DATE
DIRECTORY=./save/${DATE}/
if [ ! -d "./save/" ]; then
    mkdir ./save/
fi

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

# For reproducibility in a eGPU environment
export CUBLAS_WORKSPACE_CONFIG=:16:8

########### RUN MAIN.py ###############
dataset=mnist
model=LeNet
start=2
end=20
step=2
n_epoch=50


# dataset=cifar10
# model=ResNet18
# start=10
# end=70
# step=5
# n_epoch=150

# dataset=gtsrb
# model=ResNet18
# start=10
# end=70
# step=5
# n_epoch=150

# strategies=(
#             # 'RandomSampling' \
#             # 'CoreSet' \
#             # 'BadgeSampling' \
#             # 'BALDDropout' \
#             # 'LeastConfidence' \
#             # 'KMeansSampling' \
#             # 'AdversarialBIM' \
#             'WAAL' \
#             'ActiveLearningByLearning' \
#             'VAAL' \
#             'LearningLoss' \
#             'ClusterMarginSampling' \
#             'uncertainGCN' \
#             'coreGCN' \
#             'MCADL' \
#             # 'ssl_LC' \
#             # 'ssl_Random' \
#             # 'ssl_Diff2AugKmeans' \
#             )

strategies=(
            # 'RandomSampling' \
            # 'CoreSet' \
            # 'BadgeSampling' \
            # 'BALDDropout' \
            # 'LeastConfidence' \
            # 'KMeansSampling' \
            'AdversarialBIM' \
            # 'WAAL' \
            # 'ActiveLearningByLearning' \
            # 'VAAL' \
            # 'LearningLoss' \
            # 'ClusterMarginSampling' \
            # 'uncertainGCN' \
            # 'coreGCN' \
            # 'MCADL' \
            # 'ssl_LC' \
            # 'ssl_Random' \
            # 'ssl_Diff2AugKmeans' \
            )
            

save_path=save/${DATE}/${dataset}
save_file='main_result.csv'

data_path='/research/dept2/yuli/datasets'

for rand_idx in 1 2 3 4 5
# for rand_idx in 1 2 3 4 5
do
        for strategy in "${strategies[@]}"
        do
        
            echo $strategy
            echo $dataset
            manualSeed=$((10*$rand_idx))
            echo $manualSeed
            python main.py  --model $model \
                            --nStart $start \
                            --nEnd $end \
                            --nQuery $step \
                            --n_epoch $n_epoch \
                            --dataset $dataset \
                            --strategy $strategy \
                            --rand_idx $rand_idx \
                            --save_path $save_path \
                            --save_file $save_file \
                            --manualSeed $manualSeed \
                            --data_path $data_path 
        done
done

