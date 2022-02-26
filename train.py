import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='/mnt/weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if not args.multigpu:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)


train_dataset = WIDERDetection('/DSFD_pytorch/data/face_train.txt', mode='train')

val_dataset = WIDERDetection('/DSFD_pytorch/data/face_val.txt', mode='val')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)
val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)


min_loss = np.inf


def train():
    per_epoch_size = len(train_dataset) // args.batch_size
    start_epoch = 0
    iteration = 0
    step_index = 0

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', 2, args.model)
    net = dsfd_net

    base_weights = torch.load(args.save_folder + basenet)
    print('Load base network {}'.format(args.save_folder + basenet))
    if args.model == 'vgg':
        vgg_weights = torch.load('/mnt/weights/vgg16_reducedfc.pth')
        net.vgg.load_state_dict(vgg_weights)
    else:
        net.resnet.load_state_dict(base_weights)

    if args.cuda:
        if args.multigpu:
            net = torch.nn.DataParallel(dsfd_net)
        net = net.cuda()
        cudnn.benckmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(args.cuda)
    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)

    net.train()
    for epoch in range(start_epoch, 50):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            train_loader_len = len(train_loader)
            adjust_learning_rate(optimizer, epoch, step_index, train_loader_len)
            step_index += 1

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss.data.item()

            if iteration % 10 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                    loss_c_pal1.data.item(), loss_l_pa1l.data.item()))
                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                    loss_c_pal2.data.item(), loss_l_pa12.data.item()))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'dsfd_' + repr(iteration) + '.pth'
                torch.save(dsfd_net.state_dict(),
                           os.path.join(save_folder, file))
            iteration += 1

        val(epoch, net, dsfd_net, criterion)
        if iteration == 150000:
            break


def val(epoch, net, dsfd_net, criterion):
    net.eval()
    step = 0
    losses = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2
        losses += loss.data.item()
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(dsfd_net.state_dict(), os.path.join(
            save_folder, 'dsfd.pth'))
        min_loss = tloss
    
    torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd_{}.pth'.format(epoch)))


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 10

    if epoch >= 30:
        factor = factor + 1

    lr = 1e-3 * (0.1 ** factor)

    """Warmup"""
    if epoch < 1:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(step % 10 == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
