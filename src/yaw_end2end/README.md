## 加入yaw系数的版本
延续list指定路径，对应label指定标签的结构  
    trainlabel结构：
    label_id pitch yaw roll  
    testlabel结构：
    label_id coef_yaw    
    其中trainlabel给的是三个角度，目前只用yaw角度，过sigmoid转成系数使用;  test用作者提供的msceleb直接给了系数，直接读入  
    在selfDefine.py代码里根据list中是否带'test'进行区分
###### pretrained vs. resume:  
pretrained可以选择加载部分层 resume完全加载
当前pretrained不加载最后一层全连接，以适应不同训练数据的不同id数  
resume会根据加载的epoch把start_epoch改为对应的次数，注意adjust_lr要和epoch对应
