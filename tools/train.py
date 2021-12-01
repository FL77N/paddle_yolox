import argparse
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
import paddle
from yolox.exp import get_exp
from loguru import logger
from yolox.core import Trainer,New_Trainer

import paddle.distributed as dist

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolox_lx")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=2, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/workspace/lanxin_yolox/exps/default/yolox_x.py",
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=2, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):

    init_seeds(2)
    # set environment variables for distributed training
    #trainer = Trainer(exp, args)#用的手动iter() next()
    trainer = New_Trainer(exp, args)#用 官网推荐方法
    trainer.train()

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    dist.spawn(main, args=(exp,args,), nprocs=2,gpus='1,2') #单机多卡
    #main(exp,args)#单机单卡