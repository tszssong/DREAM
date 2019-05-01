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
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  \
    --end2end   \
    --resume ./model/resnet18_30.pth.tar \
    --img_dir /cloud_data01/zhengmeisong/TrainData/dreamData/ \
    2>&1 | tee $log_file
