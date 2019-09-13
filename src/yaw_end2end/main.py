import argparse
import os,sys,shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from ResNet import resnet18, resnet50, resnet101
from MobileNet import mobilenetv2
from selfDefine import MsCelebDataset, myDataset, CaffeCrop


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='./model', type=str)
parser.add_argument('--end2end', action='store_true',\
        help='if true, using end2end with dream block, else, using naive architecture')

best_prec1 = 0   
     
def main():
    global args, best_prec1  
    args = parser.parse_args()

    print('[img_dir:]', args.img_dir)
    print('[end2end?:]', args.end2end, '[workers:]', args.workers)
    print('[batchsize:]', args.batch_size, '[lr:]', args.lr, '[epochs:]', args.epochs)
    print('[net:]', args.arch, '[resume:]',args.resume, '[pretrained:]', args.pretrained)
    sys.stdout.flush()
    train_list_file = '../../../../../data/ms1m_emore_img/total_list.txt'
    train_label_file = '../../../../../data/ms1m_emore_img/total_label_angle.txt'
    caffe_crop = CaffeCrop('train')
    # train_dataset =  MsCelebDataset(args.img_dir, train_list_file, train_label_file, 
            # transforms.Compose([caffe_crop,transforms.ToTensor()]))
    train_list = '/data03/zhengmeisong/data/ms1m_emore_img/imgs.lst'
    # train_list = '/data03/zhengmeisong/data/ms1m_emore100/imgs.lst'
    train_dataset =  myDataset(train_list,
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
   
    class_num = train_dataset.max_label + 1
    print('class_num: ',class_num)
    
    # prepare model
    model = None
    assert(args.arch in ['resnet18','resnet50','resnet101', 'mobilenetv2', 'mobilefacenet'])
    if args.arch == 'resnet18':
        model = resnet18(pretrained=False, num_classes=class_num, end2end=args.end2end)
    if args.arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num, end2end=args.end2end)
    if args.arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num, end2end=args.end2end)
    if args.arch == 'mobilenetv2':
        model = mobilenetv2(pretrained=False, num_classes=class_num, end2end=args.end2end)
    if args.arch == 'mobilefacenet':
        model = mobilefacenet(pretrained=False, num_classes=class_num, end2end=args.end2end)
    model = torch.nn.DataParallel(model).cuda()
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        
        for key in pretrained_state_dict:
            print (key)
            if 'fc.w' in key or 'fc.b' in key:
                continue 
            model_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = False
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        sys.stdout.flush()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, yaw) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        yaw = yaw.float().cuda()
        input_var = torch.autograd.Variable(input)
        yaw_var = torch.autograd.Variable(yaw)
        target_var = torch.autograd.Variable(target)

        # compute output
        pred_score = model(input_var, yaw_var)

        loss = criterion(pred_score, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred_score.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        losses.update(loss.item(),  1)
        top1.update(prec1.item(), 1)
        top5.update(prec5.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}] lr:{lr:.7f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), 
                   lr=optimizer.param_groups[0]['lr'], 
                   loss=losses, top1=top1, top5=top5))
        sys.stdout.flush()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    #torch.save(state, full_filename)   #save checkpoint.pth.tar every epoch
    epoch_num = state['epoch']
    if epoch_num%1==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint',args.arch+'_'+str(epoch_num)))

class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    #if epoch in [int(args.epochs*0.8), int(args.epochs*0.9), int(args.epochs*0.95)]:
    if epoch in [8, 12, 20, 30]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
