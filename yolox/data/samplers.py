#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import paddle


class YoloBatchSampler(paddle.io.DistributedBatchSampler):
    def __init__(self, *args, input_dimension=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx, self.mosaic) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


class YoloBatchSampler_new(paddle.io.BatchSampler):
    def __init__(self, *args, input_dimension=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx, self.mosaic) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None