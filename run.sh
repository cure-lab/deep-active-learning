HOST=$(hostname)
echo "Current host is: $HOST"
DATE=`date +%Y-%m-%d`
echo $DATE

if [ ! -d "./save/" ]; then
    mkdir ./save/
fi

DIRECTORY=./save/${DATE}/
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
# start=0.5
# end=4
# step=0.5
# n_epoch=100

# dataset=gtsrb
# model=ResNet18
# start=0.5
# end=4
# step=0.5
# n_epoch=100


strategy='RandomSampling'

#('RandomSampling'
# 'CoreSet' \
# 'BadgeSampling' \
# 'BALDDropout' \
# 'LeastConfidence' \
# 'KMeansSampling' \
# 'AdversarialBIM' \
# 'ActiveLearningByLearning' \
# 'LearningLoss' \
# 'ClusterMarginSampling' \
# 'uncertainGCN' \
# 'coreGCN' \
# 'MCADL' \
# 'WAAL' \
# 'VAAL' \
# 'ssl_Random' \
# 'ssl_Diff2AugKmeans' \
# 'ssl_Diff2AugDirect' \
# 'ssl_Consistency')
            
            
save_file=$dataset'_result.csv'
data_path='./dataset'

for random_seed in 1
do
    save_path=save/${DATE}/$strategy
    if [ ! -d "$save_path" ]; then
        mkdir ./save/${DATE}/$strategy
    fi
    echo $strategy
    echo $dataset
    python main.py  --model $model \
                    --nStart $start \
                    --nEnd $end \
                    --nQuery $step \
                    --n_epoch $n_epoch \
                    --dataset $dataset \
                    --strategy $strategy \
                    --save_path $save_path \
                    --save_file $save_file \
                    --data_path $data_path \
                    --save_model \
                    --seed $random_seed 
                    # --lr 0.1 \ # 0.01 for ssl
done

