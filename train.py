# call: python train.py --data /fastdata/rhesse/datasets/FunnyBirds/ --model resnet50 --checkpoint_dir /data/rhesse/FunnyBirds/checkpoints --checkpoint_prefix resnet50_default --pretrained

import argparse
import os
import random
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from datasets.funny_birds import FunnyBirds
from models.resnet import resnet50
from models.vgg import vgg16
from models.ViT.ViT_new import vit_base_patch16_224

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--model', required=True,
                    choices=['resnet50', 'vgg16', 'vit_b_16'],
                    help='model architecture')
parser.add_argument('--checkpoint_dir', metavar='DIR', required=True, default=None,
                    help='path to checkpoints')
parser.add_argument('--checkpoint_prefix', type=str, required=True, default=None,
                    help='checkpoint prefix')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step_size', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrained_ckpt', type=str)
parser.add_argument('--multi_target', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #cudnn.deterministic = True
        #cudnn.benchmark = True
        #warnings.warn('You have chosen to seed training. '
        #              'This will turn on the CUDNN deterministic setting, '
        #              'which can slow down your training considerably! '
        #              'You may see unexpected behavior when restarting '
        #              'from checkpoints.')

    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        model.fc = torch.nn.Linear(2048, 50)
    elif args.model == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
        model.classifier[-1] = torch.nn.Linear(4096, 50)
    elif args.model == 'vit_b_16':
        model = vit_base_patch16_224(pretrained=args.pretrained)
        model.head = torch.nn.Linear(768, 50)
    else:
        print('Model not implemented')

    model = model.cuda(args.gpu)

        
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    

    # Data loading code
    transforms = None

    train_dataset = FunnyBirds(args.data, 'train', transform = transforms)
    test_dataset = FunnyBirds(args.data, 'test', transform = transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    for epoch in range(0, args.epochs):
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, args)
        
        scheduler.step()

        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        images = samples['image']
        target = samples['class_idx']
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        if not args.multi_target:
            loss = criterion(output, target)
        else:
            B,_,_,_ = images.shape
            params = samples['params']
            loss = 0.
            for b in range(B):
                params_single = train_loader.dataset.get_params_for_single(params, idx=b)
                part_idxs = train_loader.dataset.single_params_to_part_idxs(params_single)    
                target_classes = list(range(len(train_loader.dataset.classes)))
                for part in part_idxs.keys():
                    part_idx = part_idxs[part]
                    if part_idx == -1:
                        continue
                    for class_idx in range(len(train_loader.dataset.classes)):
                        class_spec = train_loader.dataset.classes[class_idx]
                        if part_idx != class_spec['parts'][part]:
                            try:
                                target_classes.remove(class_idx)
                            except ValueError:
                                do_nothin = 'do_nothing'

                for target_class in target_classes:
                    target_class_tensor = torch.tensor([target_class]).cuda(args.gpu, non_blocking=True)
                    
                    loss += criterion(output[b].unsqueeze(0), target_class_tensor) * 1/len(target_classes) * 1/B

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, samples in enumerate(loader):
                images = samples['image']
                target = samples['class_idx']
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (False and (len(val_loader.sampler) * -1 < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    
    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename_checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint_prefix + '_checkpoint.pth.tar')
    
    torch.save(state, filename_checkpoint)
    if is_best:
        filename_checkpoint_best = os.path.join(args.checkpoint_dir, args.checkpoint_prefix + '_checkpoint_best.pth.tar')
        torch.save(state, filename_checkpoint_best)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()