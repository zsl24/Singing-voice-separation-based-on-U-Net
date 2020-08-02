#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:42:25 2017

@author: wuyiming
"""

SR = 16000
FFT_SIZE = 1024
BATCH_SIZE = 1
PATCH_LENGTH = 512
WIN_LENGTH = 1024
H = int(WIN_LENGTH * 0.25)
ALPHA = 2

PATH_FFT = "spectro"
PATH_EVAL = "dataset/test"
PATH_TRAIN = "spectro/train"
PATH_TRAIN_wav = "dataset/train"
PATH_VAL_wav = "dataset/val"
PATH_VAL = "spectro/val"
PATH_RESULT = "result"
PATH_CHECKPOINTS = "checkpoints"
