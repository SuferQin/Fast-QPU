import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from data import ModelNet40Cls,ModelNet10Cls
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import faulthandler
faulthandler.enable()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

g_acc = 0
st_epoch = 0

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/config_ssn_cls.yaml', type=str)
parser.add_argument('--resume', action="store_true")
args = parser.parse_args()
if not args.resume:
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)

class sch_lr_func:
    def __init__(self,lr_decay,decay_step,lr_clip,base_lr) -> None:
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.lr_clip = lr_clip
        self.base_lr = base_lr
    def __call__(self, e):
        return max(self.lr_decay**(e // self.decay_step), self.lr_clip / self.base_lr)

def main(args):
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        if not args.resume: print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    args.save_path = os.path.join(args.save_path,args.config.split('/')[-1].split('.')[0])
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    
    ModelNet = ModelNet40Cls if args.num_classes==40 else ModelNet10Cls
    train_dataset = ModelNet(num_points = args.num_points, root = args.data_root, transforms=train_transforms)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers), 
        pin_memory=True
    )

    test_dataset = ModelNet(num_points = args.num_points, root = args.data_root, transforms=test_transforms, train=False)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    
    model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, 
                        relation_prior = args.relation_prior, use_xyz = True, typer=args.typer)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    # lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    lr_lbmd = sch_lr_func(args.lr_decay,args.decay_step,args.lr_clip,args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset)/args.batch_size
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.save_path,'last_model.pth')))
        info = torch.load(os.path.join(args.save_path,'last_info.pth'))
        optimizer.load_state_dict(info['opt'])
        lr_scheduler.load_state_dict(info['sch'])
        global g_acc,st_epoch
        g_acc = info['best_acc']
        st_epoch = info['epoch']+1
        print('== RESUME ==')
    
    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    

def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    global g_acc,st_epoch
    batch_count = 0
    model.train()
    for epoch in range(st_epoch,args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            
            # augmentation
            points.data = PointcloudScaleAndTranslate(points.data)
            
            optimizer.zero_grad()
            
            pred = model(points)
            target = target.view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation in between an epoch
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                validate(test_dataloader, model, criterion, args, batch_count)
        torch.save(model.state_dict(), os.path.join(args.save_path,'last_model.pth'))
        torch.save({
            'opt':optimizer.state_dict(),
            'sch':lr_scheduler.state_dict(),
            'epoch':epoch,
            'best_acc': g_acc,
        }, os.path.join(args.save_path,'last_info.pth'))


def validate(test_dataloader, model, criterion, args, iter): 
    global g_acc
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            pred = model(points)
            target = target.view(-1)
            loss = criterion(pred, target)
            losses.append(loss.data.clone())
            _, pred_choice = torch.max(pred.data, -1)
            
            preds.append(pred_choice)
            labels.append(target.data)
        
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    acc = float((preds == labels).sum()) / labels.numel()
    print('\nval loss: %0.6f \t acc: %0.6f\n' %(np.array(losses).mean(), acc))
    if acc > g_acc:
        g_acc = acc
        torch.save(model.state_dict(), os.path.join(args.save_path,'model.pth'))
        with open(os.path.join(args.save_path,'best.txt'),'a') as f: f.write(f'iter:{iter} acc:{acc}\n')

    model.train()
    
if __name__ == "__main__":
    main(args)