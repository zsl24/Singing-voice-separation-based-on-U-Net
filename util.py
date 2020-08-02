#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:47:06 2017

@author: wuyiming
"""
import chainer.links as L
from librosa.display import specshow,waveplot
from librosa.util import find_files
from librosa.core import stft, load, istft, resample
from librosa.output import write_wav
import network
import const as C
import numpy as np
from chainer import config
import os
import os.path
import pandas.core.groupby.ops
from pesq import pesq
from mir_eval.separation import bss_eval_sources
from pylab import savefig
import matplotlib.pyplot as plt




def SaveSpectrogram(y_mix, y_vocal, y_inst, fname):
    S_mix = np.abs(
        stft(y_mix, n_fft=C.FFT_SIZE, hop_length=C.H,win_length=C.WIN_LENGTH)).astype(np.float32)
    S_vocal = np.abs(
        stft(y_vocal, n_fft=C.FFT_SIZE, hop_length=C.H,win_length=C.WIN_LENGTH)).astype(np.float32)
    S_inst = np.abs(
        stft(y_inst, n_fft=C.FFT_SIZE, hop_length=C.H,win_length=C.WIN_LENGTH)).astype(np.float32)
    assert(S_mix.shape == S_vocal.shape == S_inst.shape)

    if S_mix.shape[1] < C.PATCH_LENGTH:
        diff = C.PATCH_LENGTH - S_mix.shape[1]
        S_mix = np.pad(S_mix,((0,0),(0,diff)),'constant')
        S_vocal = np.pad(S_vocal,((0,0),(0,diff)),'constant')
        S_inst = np.pad(S_inst,((0,0),(0,diff)),'constant')
    norm = S_mix.max()
    S_mix /= norm
    S_vocal /= norm
    S_inst /= norm

    np.savez(os.path.join(C.PATH_FFT, fname+".npz"),
             mix=S_mix, vocal=S_vocal, inst=S_inst)


def LoadDataset(target, pth):
    filelist_fft = os.listdir(pth)
    #filelist_fft = find_files(pth, ext="npz")
    Xlist = []
    Ylist = []
    for file_fft in filelist_fft:
        if len(Xlist)==600:
            break
        file_fft = os.path.join(pth,file_fft)
        dat = np.load(file_fft)
        Xlist.append(dat["mix"])
        if target == "vocal":
            assert(dat["mix"].shape == dat["vocal"].shape)
            Ylist.append(dat["vocal"])
        else:
            assert(dat["mix"].shape == dat["inst"].shape)
            Ylist.append(dat["inst"])
    return Xlist, Ylist


def LoadAudio(fname):#transform waveform to spectrogram
    y, sr = load(fname, sr=C.SR)
    spec = stft(y, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.WIN_LENGTH)

    if spec.shape[1] < C.PATCH_LENGTH:
        df = C.PATCH_LENGTH - spec.shape[1]
        spec = np.pad(spec,((0,0),(0,df)),'constant')
    else:
        spec = spec[:,0:C.PATCH_LENGTH]

    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase,spec


def SaveAudio(fname, spec, length):
    fname = os.path.join(C.PATH_RESULT,fname)
    wav = istft(spec, hop_length=C.H, win_length=C.WIN_LENGTH)
    crop = min(wav.shape[0],length)
    wav = wav[:crop]
    write_wav(fname, wav, C.SR, norm=True)
    return wav, crop


def ComputeMask(input_mag, input_spec, alpha, unet_model="unet.model"):
    '''
    parameter:
                input_mag - 2D numpy array of magnitude of spectrogram
                input_spec - 2D numpy array of spectrogram
                alpha - parameter of winner filtering             
                unet_model - the name of U-Net model with trained parameters
    return: pred_spec - the predicted spectrogram of singing voice or the source we want to separate
    '''   
    
    unet = network.UNet()
    unet.load(unet_model)
    num_para = unet.conv1.W.size + unet.conv1.b.size+ unet.conv2.W.size + unet.conv2.b.size+ unet.conv3.W.size + unet.conv3.b.size+ unet.conv4.W.size + unet.conv4.b.size+ unet.conv5.W.size + unet.conv5.b.size+ unet.conv6.W.size + unet.conv6.b.size
    
    config.train = False
    config.enable_backprop = False
    pred_mag = unet(input_mag[np.newaxis, np.newaxis, 1:, :]).data[0, 0, :, :] #pred_mag = (512,512)
    mask = np.power(pred_mag,alpha) / np.power(input_mag[1:,:],alpha)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.isnan(mask[i,j]):
                mask[i,j] = 0
    pred_spec = mask * input_spec[1:,:]
    pred_spec = np.vstack((np.zeros(pred_spec.shape[1], dtype="float32"), pred_spec))


    return pred_spec

def metrics(mix, inst, vocal, inst_pred, vocal_pred):
    '''
    parameter:
                mix - 1D numpy array of wavform of mixture
                inst - 1D numpy array of wavform of pure instrument sound
                vocal - 1D numpy array of wavform of pure singing voice                
                inst_pred - 1D numpy array of wavform of predicted instrument sound
                vocal_pred - 1D numpy array of wavform of predicted singing voice
    return: the performance of predict results, all are float

    function: calculate metrics of predicted results using references
    '''    
    pesq_inst = pesq(C.SR, inst, inst_pred, 'nb') 
    pesq_vocal = pesq(C.SR, vocal, vocal_pred, 'nb')
    sdr, sir, sar, _ = bss_eval_sources(np.array([inst, vocal]),
                                       np.array([inst_pred, vocal_pred]), False)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([inst, vocal]),
                                          np.array([mix, mix]), False)
    nsdr = sdr - sdr_mixed
    
    return nsdr, sir, sar, pesq_inst, pesq_vocal


def plot_spec(spec,fname,datatype):
    '''
    parameter:
                spec - 2D numpy array of spectrogram
                fname - string, filename of one song, in the form of：xxx.wav
                datatype - string, 'mix' or 'vocal_pure' or 'inst_pure'
    function: plot the spectrogram of specific audio file
    '''
    s=0.5
    fname = fname.split('.')[0]
    plt.figure(figsize=(12*s, 8*s))
    specshow(spec, x_axis = 'time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('U-Net ' + datatype + ' spectrogram:' + fname)      
    path = os.path.join('C:\\Singing_voice\\UUNet\\figure\\spectrogram',fname)
    if os.path.exists(path)==False:
        os.mkdir(path)
    fname = datatype  + '.png'
    path = os.path.join(path,fname)
    savefig(path)

def plot_wav(wav,fname,datatype):
    '''
    parameter:
                spec - 1D numpy array of wavform
                fname - string, filename of one song, in the form of：xxx.wav
                datatype - string, 'mix' or 'vocal_pure' or 'inst_pure'
    function: plot the wavform of specific audio file
    '''
    s=0.5
    fname = fname.split('.')[0]
    plt.figure(figsize=(12*s, 8*s))
    waveplot(wav, sr = C.SR, max_points=100000.0)
    plt.title('U-Net ' + datatype + ' waveform:' + fname)      
    path = os.path.join('C:\\Singing_voice\\UUNet\\figure\\waveform',fname)
    if os.path.exists(path)==False:
        os.mkdir(path)
    fname = datatype  + '.png'
    path = os.path.join(path,fname)
    savefig(path)    

    
   
    
    
    
    
    
    
    