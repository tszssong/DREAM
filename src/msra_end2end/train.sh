model_dir=model 
if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi
log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

curr_date=$(date +'%m_%d_%H_%M') 
log_file="./log/$curr_date.log"

# train the model with GPUs 0
CUDA_VISIBLE_DEVICES=0 python main.py  \
    --end2end   \
    --img_dir /media/ubuntu/9a42e1da-25d8-4345-a954-4abeadf1bd02/home/ubuntu/song/ \
    2>&1 | tee $log_file