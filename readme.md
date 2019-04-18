# DREAM block for Pose-Robust Face Recognition
This is our implementation for our CVPR 2018 accepted paper *Pose-Robust Face Recognition via Deep Residual Equivariant Mapping* [paper on arxiv](https://arxiv.org/abs/1803.00839).

The code is wriiten by [Yu Rong](https://github.com/penincillin) and [Kaidi Cao](https://github.com/CarlyleCao)

## Prerequisites
- ubuntu 16.04 配一块1080T, cuda版本9.1
- Anaconda2 + Python 2.7
- opencv3.4.1
- 按作者git下载模型和数据
## Train DREAM Block
### stitch Training
Prepare the feature extracted from any face recognition model (You could use the pretrained model we prepared).   
We prepared a piece of sample data (stitching.zip) which could be download from [Google Drive](https://drive.google.com/file/d/1x1K8MxAnVtpfaN3DfO4bdcKH39mmplj-/view?usp=sharing) &nbsp; &nbsp; [Baidu Yun](https://pan.baidu.com/s/1QIEeE9RxRY6iK3wCpvUh2Q)  
- Download the sample data
```bash
mkdir data
mv stitching.zip data
cd data
unzip stitching.zip
```
- Train the model:
```bash
cd src/stitching
sh train_stitch.sh
```


### end2end Training
- Download the Ms-Celeb-1M Subset
```bash
mkdir data
mv msceleb.zip data
cd data
unzip msceleb.zip
```
- Train the model:
```bash
cd src/end2end
sh train.sh
```
直接训显存不够，batchsize改成64训练loss直接nan，lr改成0.01可以；改为RESNET18，batchsize=128，loss可以降到1.
### evaluate CFP
- Download the CFP dataset and preprocess the image. Then download the image list for evaluation
```bash
# make sure you are in the root directory of DREAM project
mkdir data
cd src/preprocess
sh align_cfp.sh     ##这里用了test_process_align， ldd看它依赖的库，缺啥装啥
cd data/CFP
unzip CFP_protocol.zip
```
aligin_cfp.sh里几个变量绝对地址改为自己机器上的CFP开源数据集地址
- Download pretrained model
```bash
# make sure you are in the root directory of DREAM project
cd ../ 
mv model.zip data
cd data
unzip model.zip
```
- Evaluate the pretrained model on CFP dataset
```bash
# make sure you are in the root directory of DREAM project
cd src/CFP
sh eval_cfp.sh
```
CUDA_VISIBLE_DEVICES改成自己的卡0
eval_roc.py第96行一样的改为自己机器上的CFP开源数据集地址

### evaluate IJBA
- Download the IJBA dataset(contact me to get the aligned images)
```bash
# make sure you are in the root directory of DREAM project
mkdir data
mv IJBA.zip data
cd data
unzip IJBA.zip
```
- Download pretrained models (If have downloaded the models, skip this step)
```bash
# make sure you are in the root directory of DREAM project
cd ../ 
mv model.zip data
cd data
unzip model.zip
```
- Evaluate the pretrained model on IJBA dataset
```bash
# make sure you are in the root directory of DREAM project
cd src/IJBA
sh eval_ijba.sh
```

