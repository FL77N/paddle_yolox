#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import paddle
import paddle.nn as nn


def is_parallel(model):
    """check if model is in parallel mode."""
    #import apex
    parallel_type = (
        paddle.DataParallel,
        #paddle.dist
        #apex.parallel.distributed.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Layer): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        #self.ema = deepcopy(model.module if is_parallel(model) else model)
        self.ema = deepcopy(model.module)  # FP32 EMA
        self.ema.eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.stop_gradient = True

    def update(self, model):
        # Update EMA parameters
        with paddle.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype in [paddle.float16, paddle.float32, paddle.float64]:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)



# ================================================================
#
#   Author      : PaddleDetection
#   Created date:
#   Description :
#
# ================================================================

import numpy as np


class ExponentialMovingAverage():
    def __init__(self, model, decay, thres_steps=True):
        self._model = model
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    def register(self):
        self._update_step = 0
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:   # 只记录可训练参数。bn层的均值、方差的stop_gradient默认是True，所以不会记录bn层的均值、方差。
                self._shadow[name] = param.numpy().copy()

    def update(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._shadow
                new_val = np.array(param.numpy().copy())
                old_val = np.array(self._shadow[name])
                decay = min(self._decay, (1 + self._update_step) / (10 + self._update_step)) if self._thres_steps else self._decay
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[name] = new_average
        self._update_step += 1
        return decay

    def apply(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._shadow
                self._backup[name] = np.array(param.numpy().copy())
                param.set_value(np.array(self._shadow[name]))

    def restore(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._backup
                param.set_value(self._backup[name])
        self._backup = {}


# if __name__ == "__main__":
#
#     ema  = ExponentialMovingAverage(None,0.998)
#     print(ema._model)