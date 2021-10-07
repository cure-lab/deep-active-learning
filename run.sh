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
# n_epoch=200

# dataset=gtsrb

strategies=(
            # 'RandomSampling' \
            # 'CoreSet' \
            # 'BadgeSampling' \
            # 'BALDDropout' \
            # 'LeastConfidence' \
            # 'LeastConfidenceDropout' \
            # 'KMeansSampling' \
            # 'AdversarialBIM' \
            # 'WAAL' \
            # 'ActiveLearningByLearning' \
            'VAAL' \
            'LearningLoss' \
            'ClusterMarginSampling' \
            # 'uncertainGCN' \
            # 'coreGCN' \
            # 'LAL' \
            # 'MultiCritera' \
            )
            
save_path=save/${DATE}/${dataset}
save_file='main_result.csv'

for strategy in "${strategies[@]}"
do

        for rand_idx in 1 2 3 4 5
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
                            --manualSeed $manualSeed
        done
    # done
done

