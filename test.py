import util
import network
from librosa.util import find_files
from librosa.core import load,stft
import os.path
import const as C
import numpy as np
from pesq import pesq
from time import time

start = time()


PATH_MIR = C.PATH_EVAL
audiolist = find_files(PATH_MIR, ext="wav")
total_len = 0
gnsdr = gsir = gsar = np.zeros(2)
gpesq_inst = gpesq_vocal = 0
checkpoint = "checkpoints\checkpoint_54.model"
#checkpoint = "models\ck20200501UNET_34ep(Hop256).model"
i = 0
alpha = C.ALPHA
for audiofile in audiolist:
    i = i + 1
    if audiofile == audiolist[i-2]:
        continue
    fname = os.path.split(audiofile)[-1]
    
    y, _ = load(audiofile, sr=None, mono=False)
    inst = y[0, :]
    mix = y[0, :] + y[1, :]
    vocal = y[1, :] 
    
    #reconstruct predicted wave 
    mix_mag, mix_phase, mix_spec = util.LoadAudio(os.path.join(PATH_MIR,fname))
    vocal_mag = np.abs(stft(vocal, n_fft=C.FFT_SIZE,hop_length=C.H,win_length=C.WIN_LENGTH).astype(np.float32))
    vocal_mag /= np.max(vocal_mag)
    inst_mag = np.abs(stft(inst, n_fft=C.FFT_SIZE,hop_length=C.H,win_length=C.WIN_LENGTH).astype(np.float32))
    inst_mag /= np.max(inst_mag)
    
    vocal_pred_spec = util.ComputeMask(mix_mag, mix_spec, alpha, unet_model=checkpoint, hard=False) #load model from checkpoints, in order to get t-f mask
    inst_pred_spec = mix_spec - vocal_pred_spec
      
    vocal_pred,len_cropped = util.SaveAudio("vocal-%s" % fname, vocal_pred_spec,vocal.shape[0])
    inst_pred,_ = util.SaveAudio("inst-%s" % fname, inst_pred_spec,vocal.shape[0])
    vocal_pred = vocal_pred * 100
    inst_pred = inst_pred * 100
    spec_cropped = min(vocal_mag.shape[1],vocal_pred_spec.shape[1])
    #util.plot_spec(mix_mag[:,:spec_cropped],fname,datatype = 'mix')
    #util.plot_spec(np.abs(vocal_pred_spec)[:,:spec_cropped],fname,datatype = 'vocal_predicted')
    #util.plot_spec(np.abs(inst_pred_spec)[:,:spec_cropped],fname,datatype = 'inst_predicted')
    #util.plot_spec(vocal_mag[:,:spec_cropped],fname,datatype = 'vocal_pure')
    #util.plot_spec(inst_mag[:,:spec_cropped],fname,datatype = 'inst_pure')
    # compute metrics, including SAR, SIR, NSDR, PESQ for both vocal and instrument
    vocal = vocal[ :len_cropped]
    inst = inst[ :len_cropped]
    mix = mix[ :len_cropped]
    #util.plot_wav(vocal_pred,fname,datatype = 'vocal_predicted')
    #util.plot_wav(vocal,fname,datatype = 'vocal_pure')
    #util.plot_wav(mix,fname,datatype = 'mix')
    nsdr, sir, sar, pesq_inst, pesq_vocal = util.metrics(mix, inst, vocal, inst_pred, vocal_pred)
    print(fname,pesq_vocal)
    input()
    total_len = total_len + len_cropped
    gpesq_inst = gpesq_inst + len_cropped * pesq_inst
    gpesq_vocal = gpesq_vocal + len_cropped * pesq_vocal
    gnsdr = gnsdr + len_cropped * nsdr
    gsir = gsir + len_cropped * sir
    gsar = gsar + len_cropped * sar
    

#print('number of parameters:{}'.format(num_para))
gpesq_inst = gpesq_inst / total_len
gpesq_vocal = gpesq_vocal / total_len
gnsdr = gnsdr / total_len
gsir = gsir / total_len
gsar = gsar / total_len
end = time()
print("duration:{}".format(end - start))
print(checkpoint)
print("alpha:{}".format(alpha))
print("window_length:{}".format(C.WIN_LENGTH))
print("hop_length:{}".format(C.H))
print("PESQ_inst:{}\n PESQ_vocal:{}\n NSDR:{}\n SIR:{}\n SAR:{}".format(gpesq_inst, gpesq_vocal, gnsdr,gsir,gsar))

