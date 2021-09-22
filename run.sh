#!/bin/bash

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
start=100
end=10000
step=3000

n_epoch=2

# dataset=cifar10
# model=resnet
# start=1000
# end=2000
# step=1000

# dataset=gtsrb
# model=resnet
# start=200
# end=2000
# step=1800

# dataset=gtsrb
# model=resnet
# start=100
# end=3000
# step=2900


strategies=(
            # 'RandomSampling' \
            'CoreSet' \
            'BadgeSampling' \
            'BALDDropout' \
            'LeastConfidence' \
            'LeastConfidenceDropout' \
            'KMeansSampling' \
            'AdversarialBIM' \
            'WAAL' \
            'ActiveLearningByLearning' \
            'VAAL' \
            'LearningLoss4AL' \
            'ClusterMarginSampling' \
            )
            

save_path=save/${DATE}/${dataset}

for strategy in "${strategies[@]}"
do

        for rand_idx in 1
        do
            manualSeed=$((10*rand_idx))
            echo $strategy
            echo $dataset
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

