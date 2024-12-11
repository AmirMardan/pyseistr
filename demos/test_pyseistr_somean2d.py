## This is a DEMO script for 2D structure-oriented mean/smoothing filter
import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps

## Generate synthetic data
from pyseistr import gensyn
data=gensyn()
data=data[:,0::10] #or data[:,0:-1:10];
data=data/np.max(np.max(data))
np.random.seed(202122)
scnoi=(np.random.rand(data.shape[0],data.shape[1])*2-1)*0.2
dn=data+scnoi

print('size of data is (%d,%d)'%data.shape)
print(data.flatten().max(),data.flatten().min())

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

## Slope estimation
dtemp=dn*0 #dtemp is the preprocessed data
for i in range(1,dn.shape[0]+1):
    dtemp[i-1,:]=smooth(dn[i-1,:],5)

dip = ps.dip2dc(dtemp,
                verb=0)
print(dn.shape)
print(dip.flatten().max(),dip.flatten().min())

## Structural smoothing
r=2
eps=0.01
order=2
d1=ps.somean2dc(dn,  # noisy data
                dip,
                r,
                order,
                eps,
                verb=0)

## plot results
fig, ax = plt.subplots(2, 4,
                       figsize=(10, 8),
                       sharex=True,
                       sharey=True)
ax = ax.flatten()
ax[0].imshow(data,
             cmap='jet',
             clim=(-0.2, 0.2),
             aspect=0.5)
ax[0].set_title('Clean data')

ax[1].imshow(dn,cmap='jet',
             clim=(-0.2, 0.2),
             aspect=0.5)
ax[1].set_title('Noisy data')

ax[2].imshow(dtemp,
             cmap='jet',
             clim=(-0.2, 0.2),
             aspect=0.5)
ax[2].set_title('Filtered (MEAN)')

ax[3].imshow(dn-dtemp,
             cmap='jet',
             clim=(-0.2, 0.2),
             aspect=0.5)
ax[3].set_title('Noise (MEAN)')

ax[5].imshow(dip,
             cmap='jet',
             clim=(-2, 2),
             aspect=0.5)
ax[5].set_title('Slope')

ax[6].imshow(d1,
             cmap='jet',
             clim=(-0.2, 0.2),
             aspect=0.5)
ax[6].set_title('Filtered (SOMEAN)')

ax[7].imshow(dn-d1,
             cmap='jet',
             clim=(-0.2, 0.2),
             aspect=0.5)
ax[7].set_title('Noise (SOMEAN)')

ax[4].set_visible(False)
file_name = __file__.split('/')[-1].split('.')[0]
fig.suptitle(file_name)
# plt.savefig('test_pyseistr_somean2d.png',
#             format='png',
#             dpi=300)
plt.show()


