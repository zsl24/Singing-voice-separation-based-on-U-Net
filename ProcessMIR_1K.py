#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:49:54 2017

@author: wuyiming
"""

from librosa.util import find_files
from librosa.core import load
import os.path
import util
import const as C


PATH_MIR = C.PATH_VAL_wav

audiolist = find_files(PATH_MIR, ext="wav")

for audiofile in audiolist:
    fname = os.path.split(audiofile)[-1]
    print("Processing: %s" % fname)
    y, _ = load(audiofile, sr=None, mono=False)
    inst = y[0, :]
    mix = y[0, :]+y[1, :]
    vocal = y[1, :]
    util.SaveSpectrogram(mix, vocal, inst, fname)
