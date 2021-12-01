#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
#单机单卡

import datetime
import os
import time
from loguru import logger
import paddle
from paddle import amp

import paddle.distributed as dist
from visualdl import LogWriter
from yolox.utils import (
    MeterBuffer,
    load_ckpt,
    occupy_mem,
    gpu_mem_usage,
    ModelEMA,
    setup_logger,
    save_checkpoint,
)


class New_Trainer:
    """
    训练器  简易 缩短时间
    """
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        # training related attr
        self.max_epoch = exp.max_epoch
        self.is_distributed = True#dist.get_world_size() > 1
        self.rank = dist.get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = paddle.float16 if args.fp16 else paddle.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        model = self.exp.get_model()

        if self.is_distributed:
            dist.init_parallel_env()
            model = paddle.DataParallel(model)
        elif self.local_rank >=0:
            paddle.set_device("GPU:" + str(self.local_rank))

        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        model = self.resume_train(model)

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
        )
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = LogWriter(self.file_name)
        self.model = model
        self.model.train()
        logger.info("Training start...")
        for self.epoch in range(self.start_epoch, self.max_epoch):
            logger.info("---> start train epoch{}".format(self.epoch + 1))
            if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logger.info("--->No mosaic aug now!")
                self.train_loader.close_mosaic()
                logger.info("--->Add additional L1 loss now!")
                # if self.is_distributed:
                #     self.model.module.head.use_l1 = True
                # else:
                self.model.head.use_l1 = True
                self.exp.eval_interval = 1
                if not self.no_aug:
                    self.save_ckpt(ckpt_name="last_mosaic_epoch")

            for batch_id, data in enumerate(self.train_loader()):
                self.iter = batch_id
                iter_start_time = time.time()
                inps = paddle.cast(data[0], self.data_type)
                targets = paddle.cast(data[1], self.data_type)
                targets.stop_gradient = True
                data_end_time = time.time()
                outputs = self.model(inps, targets)
                loss = outputs["total_loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                if self.use_model_ema:
                    self.ema_model.update(self.model)

                lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
                self.optimizer.set_lr(lr)
                iter_end_time = time.time()
                self.meter.update(
                    iter_time=iter_end_time - iter_start_time,
                    data_time=data_end_time - iter_start_time,
                    lr=lr,
                    **outputs,
                )

                if (self.iter + 1) % self.exp.print_interval == 0:
                    # TODO check ETA logic
                    left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
                    eta_seconds = self.meter["iter_time"].global_avg * left_iters
                    eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                    progress_str = "epoch: {}/{}, iter: {}/{}".format(
                        self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
                    )
                    loss_meter = self.meter.get_filtered_meter("loss")
                    loss_str = ", ".join(
                        ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
                    )

                    time_meter = self.meter.get_filtered_meter("time")
                    time_str = ", ".join(
                        ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
                    )

                    logger.info(
                        "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                            progress_str,
                            gpu_mem_usage(self.local_rank),
                            time_str,
                            loss_str,
                            self.optimizer.get_lr()  # self.meter["lr"].latest,
                        )
                        + (", size: {:d}, {}".format(self.input_size[0], eta_str))
                    )
                    self.meter.clear_meters()

                # random resizing
                if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
                    self.input_size = self.exp.random_resize(
                        self.train_loader, self.epoch, self.rank, self.is_distributed
                    )


            self.save_ckpt(ckpt_name="latest")

            if (self.epoch + 1) % self.exp.eval_interval == 0:
                # all_reduce_norm(self.model)
                self.evaluate_and_save_model()

        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100
            )
        )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        # if self.use_model_ema:
        #     self.ema_model.apply()
        # evalmodel = self.model
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        #synchronize()

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pdparams")
            else:
                ckpt_file = self.args.ckpt

            ckpt = paddle.load(ckpt_file)
            # resume the model/optimizer state dict
            model.load_dict(ckpt)
            # self.optimizer.load_dict(ckpt["optimizer"])
            # resume the training states variables
            # if self.amp_training and "amp" in ckpt:
            #     amp.load_dict(ckpt["amp"])
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                #ckpt = paddle.load(ckpt_file)["model"]
                ckpt = paddle.load(ckpt_file)
                model = load_ckpt(model, ckpt)
                logger.info("loaded checkpoint done.")
            self.start_epoch = 0

        return model