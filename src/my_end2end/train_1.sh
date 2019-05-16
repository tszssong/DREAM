model_dir=model_1
if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi
log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

curr_date=$(date +'%m_%d_%H_%M') 
log_file="./log/1yaw_$curr_date.log"

# train the model with GPUs 0
CUDA_VISIBLE_DEVICES=1 python main.py  \
    --end2end --lr 0.1   \
    --batch-size 256 \
    --model_dir $model_dir \
    2>&1 | tee $log_file