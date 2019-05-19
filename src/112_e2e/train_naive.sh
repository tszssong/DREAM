model_dir=model_naive 
if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi
log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

curr_date=$(date +'%m_%d_%H_%M') 
log_file="./log/naive$curr_date.log"

# train the model with GPUs 0
CUDA_VISIBLE_DEVICES=3 python main.py  \
    --lr 0.1   \
    --arch resnet18 \
    --batch-size 2 \
    --model_dir $model_dir \
    2>&1 | tee $log_file
