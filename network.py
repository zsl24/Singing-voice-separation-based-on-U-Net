from chainer import Chain, serializers, optimizers, cuda, config
import chainer.links as L
import chainer.functions as F
from chainer import iterators
from chainer import Variable
import numpy as np
import const
import os.path
import pandas as pd
from time import time
import util
cp = cuda.cupy


class UNet(Chain):
    def __init__(self):
        super(UNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, 4, 2, 1)
            self.norm1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, 4, 2, 1)
            self.norm2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, 4, 2, 1)
            self.norm3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, 128, 4, 2, 1)
            self.norm4 = L.BatchNormalization(128)
            self.conv5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.norm5 = L.BatchNormalization(256)
            self.conv6 = L.Convolution2D(256, 512, 4, 2, 1)
            self.norm6 = L.BatchNormalization(512)
            
            self.deconv1 = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.denorm1 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(512, 128, 4, 2, 1)
            self.denorm2 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(256, 64, 4, 2, 1)
            self.denorm3 = L.BatchNormalization(64)
            self.deconv4 = L.Deconvolution2D(128, 32, 4, 2, 1)
            self.denorm4 = L.BatchNormalization(32)
            self.deconv5 = L.Deconvolution2D(64, 16, 4, 2, 1)
            self.denorm5 = L.BatchNormalization(16)
            self.deconv6 = L.Deconvolution2D(32, 1, 4, 2, 1)
        
    def __call__(self, X):
        h1 = F.leaky_relu(self.norm1(self.conv1(X)))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))
        h5 = F.leaky_relu(self.norm5(self.conv5(h4)))
        h6 = F.leaky_relu(self.norm6(self.conv6(h5)))
        dh = F.relu(F.dropout(self.denorm1(self.deconv1(h6))))
        dh = F.relu(F.dropout(self.denorm2(self.deconv2(F.concat((dh, h5))))))
        dh = F.relu(F.dropout(self.denorm3(self.deconv3(F.concat((dh, h4))))))
        dh = F.relu(self.denorm4(self.deconv4(F.concat((dh, h3)))))
        dh = F.relu(self.denorm5(self.deconv5(F.concat((dh, h2)))))
        dh = F.sigmoid(self.deconv6(F.concat((dh, h1))))
        dh = dh * X
        return dh

        

    def load(self, fname="unet.model"):
        serializers.load_npz(fname, self)

    def save(self, fname="unet.model"):
        serializers.save_npz(fname, self)


class UNetTrainmodel(Chain):
    def __init__(self, unet):
        super(UNetTrainmodel, self).__init__()
        with self.init_scope():
            self.unet = unet

    def __call__(self, X, Y):
        O = self.unet(X)
        self.loss = F.mean_absolute_error(O, Y)
        return self.loss

def TrainUNet(Xlist, Ylist, Plist, Qlist, epoch=40, savefile="unet.model",checkpoint = ''):
    assert(len(Xlist) == len(Ylist))
    unet = UNet()
    if checkpoint != '':
        unet.load(checkpoint)
    model = UNetTrainmodel(unet)
    model.to_gpu(0)
    opt = optimizers.Adam()
    opt.setup(model)
    config.train = True
    config.enable_backprop = True
    itemcnt = len(Xlist)
    itemcnt_val = len(Plist)
    itemlength = [x.shape[1] for x in Xlist] 
    print('batch_size:{}'.format(const.BATCH_SIZE))
    subepoch = sum(itemlength) // const.PATCH_LENGTH // const.BATCH_SIZE * 4
    #subepoch = itemcnt // const.BATCH_SIZE 
    print("subepoch:{}".format(subepoch))
    print("ready to train")
    loss_dataframe = {
    'FFT_size':const.FFT_SIZE,
    'Hop_size':const.H,
    'Window_length':const.WIN_LENGTH,
    'Batch_size':const.BATCH_SIZE,
    'Patch_length':const.PATCH_LENGTH,
    'train_loss':[],
    'val_loss':[],
    'epoch':[]
    }    
    for ep in range(1,epoch):
        start = time()
        print("*****************************************************************")
        sum_loss = 0.0
        loss_val = 0.0
        for subep in range(subepoch):
            X = np.zeros((const.BATCH_SIZE, 1, const.FFT_SIZE//2, const.PATCH_LENGTH),
                         dtype="float32")
            Y = np.zeros((const.BATCH_SIZE, 1, const.FFT_SIZE//2, const.PATCH_LENGTH),
                         dtype="float32")
            P = np.zeros((const.BATCH_SIZE, 1, const.FFT_SIZE//2, const.PATCH_LENGTH),
                         dtype="float32")
            Q = np.zeros((const.BATCH_SIZE, 1, const.FFT_SIZE//2, const.PATCH_LENGTH),
                         dtype="float32")                                    
            idx_item = np.random.randint(0, itemcnt, const.BATCH_SIZE)
            idx_item_val = np.random.randint(0, itemcnt_val, const.BATCH_SIZE)
            for i in range(const.BATCH_SIZE):#To generate input X and Y in training set, and P and Q in validation set, both in mini-batch. 
                if itemlength[idx_item[i]] > const.PATCH_LENGTH:                    
                    randidx = np.random.randint(
                        itemlength[idx_item[i]]-const.PATCH_LENGTH)
                    X[i, 0, :, :] = \
                        Xlist[idx_item[i]][1:, randidx:randidx+const.PATCH_LENGTH]
                    Y[i, 0, :, :] = \
                        Ylist[idx_item[i]][1:, randidx:randidx+const.PATCH_LENGTH]
                else:
                    dff = const.PATCH_LENGTH - itemlength[idx_item[i]]
                    x_spec = Xlist[idx_item[i]][1:, :]
                    y_spec = Ylist[idx_item[i]][1:, :]
                    x_spec = np.pad(x_spec,((0,0),(0,dff)),'constant')
                    y_spec = np.pad(y_spec,((0,0),(0,dff)),'constant')                    
                    X[i, 0, :, :] = \
                        x_spec
                    Y[i, 0, :, :] = \
                        y_spec
                if Plist[idx_item_val[i]].shape[1] >const.PATCH_LENGTH:
                    randidx = np.random.randint(
                        Plist[idx_item_val[i]].shape[1]-const.PATCH_LENGTH)
                    P[i, 0, :, :] = \
                        Plist[idx_item_val[i]][1:, randidx:randidx+const.PATCH_LENGTH]
                    Q[i, 0, :, :] = \
                        Qlist[idx_item_val[i]][1:, randidx:randidx+const.PATCH_LENGTH]
                else:
                    dff = const.PATCH_LENGTH - Plist[idx_item_val[i]].shape[1]
                    x_spec = Plist[idx_item_val[i]][1:, :]
                    y_spec = Qlist[idx_item_val[i]][1:, :]
                    x_spec = np.pad(x_spec,((0,0),(0,dff)),'constant')
                    y_spec = np.pad(y_spec,((0,0),(0,dff)),'constant')
                    P[i, 0, :, :] = \
                        x_spec
                    Q[i, 0, :, :] = \
                        y_spec
            opt.use_cleargrads(use = True)
            opt.update(model, cp.asarray(X), cp.asarray(Y))#update parameters and compute loss for each batch
            sum_loss += model.loss.data * const.BATCH_SIZE #model.loss returns a chainer varible, model.loss.data returns the data array of the varible
            P = cp.asarray(P)
            O = unet(P)    
            loss_val_batch = F.mean_absolute_error(O ,cp.asarray(Q))
            loss_val = loss_val + loss_val_batch.data * const.BATCH_SIZE
        sum_loss = sum_loss / subepoch
        loss_val = loss_val / subepoch
        loss_dataframe['train_loss'].append(sum_loss)
        loss_dataframe['val_loss'].append(loss_val)
        loss_dataframe['epoch'].append(ep)            

        sf = os.path.join(const.PATH_CHECKPOINTS,"checkpoint_" + str(ep) + ".model")
        unet.save(sf)
        end = time()
        print("duration:{:.2f}s".format(end - start))
        print("epoch: %d/%d  loss=%.3f" % (ep, epoch, sum_loss))
        print("loss_val:{},epoch:{}".format(loss_val, ep))
    frame = pd.DataFrame(loss_dataframe)
    frame.to_excel(excel_writer = 'C:\\Singing_voice\\UUNet\\loss.xlsx',engine = 'xlsxwriter')
     