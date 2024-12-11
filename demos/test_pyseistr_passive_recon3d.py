## This is a DEMO script for 3D structure-oriented interpolation

import numpy as np
import os

import pyseistr as ps
import matplotlib.pyplot as plt

## Download data from https://github.com/aaspip/data
# https://github.com/aaspip/data/blob/main/blast_150_13_13_4ms.bin
data_path = os.path.abspath("./pyseistr/data/blast_150_13_13_4ms.bin")
fid=open(data_path,"rb")
d = np.fromfile(fid, dtype = np.float32, count = 150*169).reshape([150,169],order='F')
d=d.reshape(150,13,13,order='F')
# d = np.transpose(np.stack((d[:, :, 2], d[:, :, 2])),
#                   (1, 2, 0))


d0=d

## Create the mask (sampling operator)
[nt, n_traces, n_shots]=d.shape
mask=np.zeros([nt * n_traces * n_shots,1])
inds=np.argwhere(abs(d0.flatten(order='F'))>0.00001);
mask[inds]=1
mask=mask.reshape(nt, n_traces, n_shots,
                  order='F')

## 3D slope calculation (inline and xline)

[dipi,dipx] = ps.dip3dc(d0,mask=mask,
                        niter=10,
                        rect=[10, 5, 5]) #[dipi,dipx] = ps.dip3dc(d0); is the one used in the paper

## 3D structure-oriented interpolation
d1=ps.soint3dc(d0,mask,dipi,dipx,order=2,niter=20,njs=[1,1],drift=0,verb=1)
fig = plt.figure(figsize=(5, 8))
ax=plt.subplot(5,1,1)
plt.imshow(d0.reshape(nt, n_traces*n_shots,order='F'),
           cmap='jet',
           clim=(-0.001, 0.001),
           aspect=0.25)
ax.set_xticks([])
ax.set_yticks([])
#or ax.set_xticks([]);ax.set_yticks([]);
ax.xaxis.set_tick_params(labelsize=6)
ax.yaxis.set_tick_params(labelsize=6)
plt.title('Raw incomplete passive data',fontsize=10)

ax=plt.subplot(5,1,2)
plt.imshow(dipi.reshape(nt, n_traces*n_shots,order='F'),
           cmap='jet',clim=(-2, 2),aspect=0.25)
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_tick_params(labelsize=6)
ax.yaxis.set_tick_params(labelsize=6)
plt.title('Iline slope',fontsize=10)

ax=plt.subplot(5,1,3)
plt.imshow(dipx.reshape(nt, n_traces*n_shots,order='F'),
           cmap='jet',
           clim=(-2, 2),
           aspect=0.25)
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_tick_params(labelsize=6)
ax.yaxis.set_tick_params(labelsize=6)
plt.title('Xline slope',fontsize=10)

ax=plt.subplot(5,1,4)
plt.imshow(mask.reshape(nt, n_traces*n_shots,order='F'),
           cmap='jet',
           clim=(-1, 1),
           aspect=0.25)
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_tick_params(labelsize=6)
ax.yaxis.set_tick_params(labelsize=6)
plt.title('Mask',fontsize=10)

ax=plt.subplot(5,1,5)
plt.imshow(d1.reshape(nt, n_traces*n_shots,order='F'),
           cmap='jet',
           clim=(-0.001, 0.001),
           aspect=0.25)
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_tick_params(labelsize=6)
ax.yaxis.set_tick_params(labelsize=6)
plt.title('Interpolated passive data',fontsize=10)

# plt.savefig('test_pyseistr_passive_recon3d.png',format='png',dpi=300)
plt.show()



