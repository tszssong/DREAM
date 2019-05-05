## 带yaw的版本
###### pretrained vs. resume:  
pretrained可以选择加载部分层 resume完全加载
当前pretrained不加载最后一层全连接，以适应不同训练数据的不同id数  
resume会根据加载的epoch把start_epoch改为对应的次数，注意adjust_lr要和epoch对应
