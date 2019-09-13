import argparse
import os,sys,shutil
import time
import struct as st
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from selfDefine import myDataset, CaffeCrop
from ResNet import resnet18, resnet50, resnet101

from PIL import Image
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18/checkpoint_40.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model', type=str)


def extract_feat(arch, resume):
    global args, best_prec1
    args = parser.parse_args()
    
    if arch.find('end2end')>=0:
        end2end=True
    else:
        end2end=False

    arch = arch.split('_')[0]

    # load data and prepare dataset
    list_file = '/home/ubuntu/zms/data/ms1m_emore100/imgs.lst'
    caffe_crop = CaffeCrop('test')
    trans = transforms.Compose([caffe_crop,transforms.ToTensor()])
    test_dataset = myDataset( list_file, transforms.Compose([caffe_crop,transforms.ToTensor()]))

    class_num = 12
    # class_num = 85742
    
    model = None
    assert(arch in ['resnet18','resnet50','resnet101'])
    if arch == 'resnet18':
        model = resnet18(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)
    if arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num,\
                extract_feature=True, end2end=end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num,\
                extract_feature=True, end2end=end2end)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    assert(os.path.isfile(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True
    
    data_num = len(test_dataset)
    feat_dim = 256
    with open(list_file, 'r') as imf:
        lines = imf.readlines()
        for line in lines:
            img_path, label = line.strip().split(' ')
            label = int(label)
            info_path = img_path.replace('.jpg', '.info')
            with open(info_path, 'r') as infof:
                infos = infof.readlines()
            yaw = infos[0].strip().split(',')[1]
            yaw = float(yaw)
            yaw = np.array(yaw).reshape(1,-1)
            img = Image.open(img_path).convert("RGB")
            img = trans(img)
            img = img.resize(1,3,112,112)
            img = img.cuda()
            # coef_yaw = torch.FloatTensor(yaw)

            yaw = torch.from_numpy(yaw)
            yaw = yaw.float().cuda()
            input_var = torch.autograd.Variable(img, volatile=True)
            yaw_var = torch.autograd.Variable(yaw, volatile=True)
            output = model(input_var, yaw_var)

            output_data = output.cpu().data.numpy()
            feat_num  = output.size(0)
            print(feat_num)
            print(output_data)

if __name__ == '__main__':
    
    #infos = [ ('resnet50_naive', '../../data/model/cfp_res50_naive.pth.tar'), 
     #         ('resnet50_end2end', '../../data/model/cfp_res50_end2end.pth.tar'), ]

    #infos = [ ('resnet18_naive', '../yaw_end2end/model/resnet18_%d.pth.tar'%i) for i in range(1,81)]
    infos = [ ('resnet18_naive', './model_naive/resnet18_%d.pth.tar'%i) for i in range(1,81)]

    for arch, model_path in infos:
        print("{} {}".format(arch, model_path))
        extract_feat(arch, model_path)
        # eval_roc_main()
        print()
