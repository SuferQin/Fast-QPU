from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
from pointnet2.utils.qpu_layers import QuaterPostProcess
import pprint
import os.path as osp
import os
import sys
import argparse

# from pointnet2.models import Pointnet2ClsMSG as Pointnet


from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import ModelNet40Cls, ModelNet10Cls
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
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate") # 1e-2
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=2e5, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-logfile", type=str, default='default', help="Redirect output to file, also set path for the saved checkpoint"
    )
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-typer", type=str, default=None, nargs='+', choices=QuaterPostProcess.TYPE_LEN.keys(), help=f"real parts mapping function choices={QuaterPostProcess.TYPE_LEN.keys()}"
    )
    parser.add_argument(
        "-mn", type=int, default=40, help="choose from [10, 40]"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    parser.add_argument('-debug',action='store_true')
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom", action="store_true")

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python -m pointnet2.train.train_cls -logfile mn40_RI -typer theta inner
    args = parse_args()
    if args.typer is None:
        from pointnet2.models import Pointnet2ClsSSG as Pointnet
    else:
        print(f'\033[33m ############## Type R: {args.typer} ######################## \033[0m')
        from pointnet2.models import Pointnet2ClsSSGQPU as Pointnet
    
    save_base = f'../../logs/qpn++/{args.logfile}'
    ckp_path = f"{save_base}/checkpoints/"
    if not osp.isdir(ckp_path):os.makedirs(ckp_path)

    if not args.debug:
        key=''
        logname = os.path.join(save_base, 'log.txt')
        if os.path.exists(logname):
            key = input('Continue? y/n (default n):')
            if key not in ['y','t']:
                raise
        if key!='t':
            sys.stdout=open(logname, 'w')
    

    train_transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            # d_utils.PointcloudRotate(),
            # d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )
    test_transforms = transforms.Compose(
        [d_utils.PointcloudToTensor(),]
    )

    ModelNetCls = ModelNet40Cls if args.mn == 40 else ModelNet10Cls
    test_set = ModelNetCls(args.num_points, transforms=test_transforms, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    train_set = ModelNetCls(args.num_points, transforms=train_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = Pointnet(input_channels=0, num_classes=args.mn, use_xyz=False, type_r=args.typer)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda it: max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        lr_clip / args.lr,
    )
    bn_lbmd = lambda it: max(
        args.bn_momentum
        * args.bnm_decay ** (int(it * args.batch_size / args.decay_step)),
        bnm_clip,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    if args.visdom:
        viz = pt_utils.VisdomViz(port=args.visdom_port)
    else:
        viz = pt_utils.CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name=os.path.join(ckp_path, args.logfile),
        best_name=os.path.join(ckp_path, args.logfile) + '_best',
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        viz=viz,
    )

    trainer.train(
        it, start_epoch, args.epochs, train_loader, test_loader, best_loss=best_loss
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(test_loader)
