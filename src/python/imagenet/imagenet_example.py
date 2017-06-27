import argparse
import os
import shutil
import time

import pandas
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
from torch.nn.utils.clip_grad import clip_grad_norm

from netbatch_dataset import NetbatchImageDataset, NetBatchSource

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


best_prec1 = 0
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='squeezenet1_1',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--sub', default='ipc:///tmp/imgpipe.sock', metavar='SUB',
                    help='Netbatch subscription URL - default: ipc:///tmp/imgpipe.sock')
parser.add_argument('--req', default='tcp://127.0.0.1:9876', metavar='REQ',
                    help='Netbatch request URL - default: tcp://127.0.0.1:9876')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=160, type=int,
                    metavar='N', help='mini-batch size (default: 160 )')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

def main():
    global args, best_prec1

    args = parser.parse_args()
    # create model
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=False)
    #
    # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()
    from imagenet import resnet_prelu
    model = torch.nn.DataParallel(resnet_prelu.resnet18()).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=0.00001, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)

    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
   # optionally resume from a checkpoint
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

    # Data loading code
    traindir = 'train-recs/'
    valdir = 'validation-recs/'
    netbatch_source = NetBatchSource(start_batch_id=int(time.time()*1000), req_url = args.req, sub_url=args.sub )
    netbatch_source.connect()
    netbatch_source.start_sub()

    train_loader = NetbatchImageDataset(netbatch_source)
    val_loader = NetbatchImageDataset(netbatch_source)

    netbatch_source.add_receiver(train_loader)
    netbatch_source.add_receiver(val_loader)
    print("Listing Training Records")
    train_indexfiles = netbatch_source.query_files(traindir, ".idx")
    #print("Listing Validation Records")
    #val_indexfiles = netbatch_source.query_files(valdir, ".idx")
    tset = set([ti[len(traindir):] for ti in train_indexfiles])
    #vset = set([ti[len(valdir):] for ti in val_indexfiles])
    #assert(tset==vset)
    for i, path in enumerate(sorted(train_indexfiles.keys())):
        rcount = int(train_indexfiles[path] / 8)
        train_loader.register_recordfile(path[:-4], rcount, 1.0, i)

    #for i, path in enumerate(sorted(val_indexfiles.keys())):
    #    rcount = int(val_indexfiles[path] / 8)
    #    val_loader.register_recordfile(path[:-4], rcount, 1.0, i)
    print("Pre-Requesting Training Batches")
    train_loader.set_batchsize(args.batch_size)
    train_loader.request_batch(3)
    #print("Pre-Requesting Validation Batch")
    #val_loader.set_batchsize(args.batch_size)
    #val_loader.request_batch(1)
    val_loader = train_loader
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("Starting Epoch %d" % (epoch))
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, 2)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, step_every_n=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    optimizer.zero_grad()
    starttime = time.time()
    end = time.time()
    lt = time.time()
    nbatches = 1000000//args.batch_size
    grad_l2_norms = list()
    grad_inf_norms = list()
    l2maxnorm = 10.0
    infmaxnorm = 1.0
    for i in range(nbatches):
        now = time.time()
        btime = lt-now
        lt = now
        (input, target) = train_loader.next_batch()
        train_loader.request_batch()

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step

        loss.backward()
        if (i % step_every_n == 0):
            if (len(grad_l2_norms)>25):
                grad_l2_norms[-25:]
            grad_l2_norm = clip_grad_norm(model.parameters(), l2maxnorm, 2)
            grad_inf_norm = clip_grad_norm(model.parameters(), infmaxnorm, float('inf'))
            grad_l2_norms.append(grad_l2_norm)
            grad_inf_norms.append(grad_inf_norm)
            print("Grad Norms: L2=%f, Inf: %f" % (grad_l2_norm, grad_inf_norm))
            optimizer.step()
            optimizer.zero_grad()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        ttime = time.time()-starttime
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'ttime {ttime:.3f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, nbatches, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, ttime=ttime))
            l2n = pandas.Series(grad_l2_norms)
            print("Gradient L2-Norm Quantiles: Max: %f, 80%%: %f - 50%%: %f" % (l2n.max(), l2n.quantile(0.8), l2n.quantile(0.5)))
            lin = pandas.Series(grad_inf_norms)
            print("Gradient Inf-Norm Quantiles: Max: %f, 80%% %f - 50%%t: %f" % (lin.max(), lin.quantile(0.8), lin.quantile(0.5)))
            if (len(grad_l2_norms)>=10):
                l2maxnorm = l2n.quantile(0.8)
                infmaxnorm = lin.quantile(0.8)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    lt = time.time()
    for i in range(50):
        now = time.time()
        print("VAL BATCH %.3f" % (lt-now))
        lt = now
        (input, target) = val_loader.next_batch()
        val_loader.request_batch()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i,1000, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    lr = args.lr * (0.1 ** (epoch // 30       ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
