#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:52:48 2017

@author: wuyiming
"""

import util
import network
import const as C

Xlist,Ylist = util.LoadDataset("vocal", C.PATH_TRAIN) 
#Xlist contains the spectrograms of mixture version of song clips in training set, 
#Ylist contains the spectrogram of pure singing voice version of song clips in training set.
Plist,Qlist = util.LoadDataset("vocal", C.PATH_VAL) 
#Plist contains the spectrograms of mixture version of song clips in validation set, 
#Ylist contains the spectrogram of pure singing voice version of song clips in validation set.
print("Dataset loaded.")
ckp = "models\ck20200501UNET_34ep(Hop256).model" #checkpoint if you want reload the mode from checkpoint, instead of starting from the beginning.
network.TrainUNet(Xlist,Ylist,Plist,Qlist,savefile="unet.model",checkpoint = ckp,epoch=60)

