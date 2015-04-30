
from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
%precision 4
plt.style.use('ggplot')

import fastclime as fc
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import matplotlib.cm as cm

fastclime = importr('fastclime')
flare = importr('flare')
grdevices = importr('grDevices')
base = importr('base')
stats = importr('stats')

L = flare.sugm_generator(n = 200, d = 60,graph="band",g=10,seed=1234)
x = np.array(L.rx2('data'))
fcout = fc.fastclime_R(x)
fc_res = fc.fastclime_select(x,fcout.lambdamtx,fcout.icovlist)
flareout = flare.sugm(L.rx2('data'),np.sqrt(np.log(60)/200),method="tiger",prec=1e-5,standardize=True)


fig = plt.figure(figsize=(12,6))
fig.add_subplot(131)
plt.imshow(np.array(L.rx2('omega')),cmap = cm.Greys_r)
plt.title('Truth')
plt.axis('off')
fig.add_subplot(132)
plt.imshow(fc_res.opt_icov,cmap = cm.Greys_r)
plt.title('fastclime')
plt.axis('off')
fig.add_subplot(133)
plt.imshow(np.array(flareout.rx2('icov')[0]),cmap = cm.Greys_r)
plt.title('TIGER')
plt.axis('off')

plt.savefig('banded.png')
#plt.show()

from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
%precision 4
plt.style.use('ggplot')

import fastclime as fc
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import matplotlib.cm as cm

fastclime = importr('fastclime')
flare = importr('flare')
grdevices = importr('grDevices')
base = importr('base')
stats = importr('stats')

L = flare.sugm_generator(n = 200, d = 60,graph="random",seed=1234)
x = np.array(L.rx2('data'))
fcout = fc.fastclime_R(x)
fc_res = fc.fastclime_select(x,fcout.lambdamtx,fcout.icovlist)
flareout = flare.sugm(L.rx2('data'),np.sqrt(np.log(60)/200),method="tiger",prec=1e-5,standardize=True)

fig = plt.figure(figsize=(12,6))
fig.add_subplot(131)
plt.imshow(np.array(L.rx2('omega')),cmap = cm.Greys_r)
plt.title('Truth')
plt.axis('off')
fig.add_subplot(132)
plt.imshow(fc_res.opt_icov,cmap = cm.Greys_r)
plt.title('fastclime')
plt.axis('off')
fig.add_subplot(133)
plt.imshow(np.array(flareout.rx2('icov')[0]),cmap = cm.Greys_r)
plt.title('TIGER')
plt.axis('off')

plt.savefig('random.png')