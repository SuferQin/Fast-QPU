from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import argparse

# from pointnet2.models import Pointnet2ClsMSG as Pointnet
# from pointnet2.models import Pointnet2ClsMSGFC as Pointnet

from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import ModelNet40Cls,ModelNet10Cls
import pointnet2.data.data_utils as d_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=1024, help="Number of points to train with"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-r","--rotate", action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.checkpoint)
    with open(os.path.join(args.checkpoint.rsplit('/',1)[0],'log.txt')) as f:
        line = f.readline()
        while '{' not in line: 
            line = f.readline()
        while '}' not in line:
            line += f.readline()
    info=eval(line)
    info['mn'] = info.get('mn',40)
    print(info)
    print(f'\033[33m Rotate: {args.rotate} \033[0m')
    if info.get('typer',None) is None:
        from pointnet2.models import Pointnet2ClsSSG as Pointnet
    else:
        print(f"\033[33m ############## Type R: {info['typer']} ######################## \033[0m")
        from pointnet2.models import Pointnet2ClsSSGQPU as Pointnet
    
    trans = [
        d_utils.PointcloudToTensor(),
        # d_utils.PointcloudScale()
    ]
    if args.rotate:
        trans.append(d_utils.PointcloudArbRotate())
    trans += [
            # d_utils.PointcloudRotate(),
            # d_utils.PointcloudRotatePerturbation(),
            # d_utils.PointcloudTranslate(),
            # d_utils.PointcloudJitter(),
            # d_utils.PointcloudRandomInputDropout()
        ]
    trans = transforms.Compose(trans)

    ModelNetCls = ModelNet40Cls if info['mn'] == 40 else ModelNet10Cls
    test_set = ModelNetCls(args.num_points, transforms=trans, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = Pointnet(input_channels=0, num_classes=info['mn'], use_xyz=False, type_r=info.get('typer',None))
    model.cuda()
    optimizer = optim.Adam(
        model.parameters()
    )

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.replace('.pth.tar','')
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer
    )
    accs = []
    for i in range(5):
        loss, stat = trainer.eval_epoch(test_loader)
        acc_per_cls = stat['acc']
        loss_per_cls = stat['loss']
        accs.append(torch.mean(torch.Tensor(acc_per_cls)).item())
        # print('loss:', loss)
        print('Average acc:', accs[-1])
    print(f'AVE5:{torch.Tensor(accs).mean()*100},{torch.Tensor(accs).std()*100}')
