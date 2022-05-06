
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from matplotlib.colors import DivergingNorm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import cos, sin, cosh, sinh

plt.rcParams.update({'font.size': 13})

def exact_data():
    data = np.loadtxt('../data/data.txt')
    return data[:, :1], data[:, 1:2], data[:, 2:] 

def sensor_data(res=20):
    data = np.loadtxt('../data/data.txt').reshape([300, 300, 3])
    idx = np.arange(300/res/2, 300, 300/res).astype('int')
    data = data[:, idx, :].reshape([300*res, 3])
    return data[:, :1], data[:, 1:2], data[:, 2:]

def noise_data(data, snr):
    sig_avg_watts = np.mean(data ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - snr
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg_watts), data.shape)
    return data + noise

snr = 10
res = 5
idx = np.arange(300/res/2, 300, 300/res).astype('int')
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 0.3])
layers = [2, 100, 100, 100, 1]

xd, td, wd = sensor_data(res)
wd = noise_data(wd, snr)

Exact_w = lambda x, t: np.sin(np.pi*x) * np.cos(2*np.pi**2*t)
Exact_w_t = lambda x, t: - 2*np.pi**2 * np.sin(np.pi*x) * np.sin(2*np.pi**2*t)

W_pred_d = np.load("phsycis-informed/w_pred_d.npy")
W_pred_p = np.load("phsycis-informed/w_pred_p.npy")
W_exact = np.load("phsycis-informed/w_exact.npy")
W_d = np.load("phsycis-informed/w_d.npy")

x = np.linspace(lb[0], ub[0], 300)[:,None]
t = np.linspace(lb[1], ub[1], 300)[:,None]

W_sig = W_d[:,idx]
"""
fig, ax = plt.subplots()
ax.plot(x, W_sig)
plt.show()
"""

############################# Plotting ###############################
fig = plt.figure(figsize=[10.4, 8.4], constrained_layout=False)
gs = gridspec.GridSpec(4, 6, figure=fig)
labelsize = 12
titlesize = 15

"""
####### Row 0,0: w(t,x) Contour plot ##################
ax = fig.add_subplot(gs[0, :3])

w = ax.imshow(W_d.T, interpolation='nearest', cmap='YlGnBu', 
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', vmin=-np.amax(np.abs(wd)), vmax=np.amax(np.abs(wd)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(w, cax=cax)
cb.ax.tick_params(labelsize=labelsize)

ax.set_xlabel('$t$', fontsize = titlesize)
ax.set_ylabel('$x$', fontsize = titlesize)
ax.set_title(r'Noisy measurement $\tilde{w}(x,t)$, SNR = %s dB, %s sensors' % (snr,res), fontsize = titlesize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
"""
"""
####### Row 0,0: w(t,x) 5 juxtaposed signals ##################
ax = fig.add_subplot(gs[0, :3])
gs00 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, :3], hspace=0)

for i in range(5):
    axs = fig.add_subplot(gs00[i])
    axs.plot(x, W_sig[:,i])
    axs.set_ylim([-np.amax(np.abs(W_sig)), np.amax(np.abs(W_sig))])
    axs.yaxis.set_major_locator(plt.MaxNLocator(3))

ax.set_title(r'Noisy measurement $\tilde{w}(x,t)$, SNR = %s dB, %s sensors' % (snr,res), fontsize = titlesize, pad=10)
ax.set_xticks([]); ax.set_yticks([])

ax.set_xlabel('$t$', fontsize = titlesize)
ax.set_ylabel('$x$', fontsize = titlesize)
ax.set_title(r'Noisy measurement $\tilde{w}(x,t)$, SNR = %s dB, %s sensors' % (snr,res), fontsize = titlesize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
"""
####### Row 0,0: w(t,x) lines ##################
ax = fig.add_subplot(gs[0, :3])

w = ax.imshow(W_sig.T, interpolation=None, cmap='YlGnBu', 
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', vmin=-np.amax(np.abs(W_sig)), vmax=np.amax(np.abs(W_sig)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(w, cax=cax)
cb.ax.tick_params(labelsize=labelsize)

line = np.linspace(t.min(), t.max(), 2)[:,None]
line2 = np.linspace(x.min(), x.max(), 6)
ax.plot(line, line2*np.ones((2,1)), 'w-', linewidth = 14)

ax.set_xlabel('$t$', fontsize = titlesize)
ax.set_ylabel('$x$', fontsize = titlesize)
ax.set_title(r'Noisy measurement $\tilde{w}(x,t)$, SNR = %s dB, %s sensors' % (snr,res), fontsize = titlesize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=labelsize)

####### Row 0,1: W_exact(t,x) ##################
ax = fig.add_subplot(gs[0, 3:])

w = ax.imshow(W_exact.T, interpolation='nearest', cmap='YlGnBu', 
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', vmin=-np.amax(np.abs(W_exact)), vmax=np.amax(np.abs(W_exact)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(w, cax=cax)
cb.ax.tick_params(labelsize=labelsize)

ax.set_xlabel('$t$', fontsize = titlesize)
ax.set_ylabel('$x$', fontsize = titlesize)
ax.set_title('Exact $w(x,t)$', fontsize = titlesize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=labelsize)

####### Row 1,0: dW(t,x) ##################
ax = fig.add_subplot(gs[1, :3])

w = ax.imshow(W_pred_d.T, interpolation='nearest', cmap='YlGnBu',
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', norm=DivergingNorm(0.), vmin=-np.amax(np.abs(W_pred_d)), vmax=np.amax(np.abs(W_pred_d)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(w, cax=cax)
cb.ax.tick_params(labelsize=labelsize)

line = np.linspace(x.min(), x.max(), 2)[:,None]
tsp = [0, 100, 200]
ax.plot(t[tsp[0]]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[tsp[1]]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[tsp[2]]*np.ones((2,1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$t$', fontsize = titlesize)
ax.set_ylabel('$x$', fontsize = titlesize)
ax.set_title('Result from data-driven method', fontsize = titlesize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=labelsize)

####### Row 1,1: FW(t,x) ##################
ax = fig.add_subplot(gs[1, 3:])

w = ax.imshow(W_pred_p.T, interpolation='nearest', cmap='YlGnBu', 
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', norm=DivergingNorm(0.), vmin=-np.amax(np.abs(W_pred_p)), vmax=np.amax(np.abs(W_pred_p)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(w, cax=cax)
cb.ax.tick_params(labelsize=labelsize)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[tsp[0]]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[tsp[1]]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[tsp[2]]*np.ones((2,1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$t$', fontsize = titlesize)
ax.set_ylabel('$x$', fontsize = titlesize)
ax.set_title('Result from physics-informed learning', fontsize = titlesize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=labelsize)

####### Row 2: w(t,x) slices ################## 
titletxt = '$t = {:.2f}$ \n Error: {:.2f}\%(DD) vs {:.2f}\%(PI)'
ax4 = fig.add_subplot(gs[2:4, 0:2])
ax4.plot(xd[:res],W_d[tsp[0],idx], 'kx', markersize=10, label = 'Noisy')   
ax4.plot(x,W_exact[tsp[0],:], 'b-', linewidth = 3, label = 'Exact')    
ax4.plot(x,W_pred_d[tsp[0],:], 'c--', linewidth = 2, label = 'Data-driven')
ax4.plot(x,W_pred_p[tsp[0],:], 'r--', linewidth = 2, label = 'Physics-informed')
ax4.set_xlabel('$x$', fontsize = titlesize); ax4.set_ylabel('$w(x,t)$', fontsize = titlesize)
errord = np.amax(W_pred_d[tsp[0],:]-W_exact[tsp[0],:]) / np.amax(W_exact) * 100
errorp = np.amax(W_pred_p[tsp[0],:]-W_exact[tsp[0],:]) / np.amax(W_exact) * 100
ax4.set_title(titletxt.format(float(t[tsp[0]]), errord, errorp), fontsize = titlesize, pad=10)
ax4.set_aspect(0.5)
ax4.set_xlim([-0.1,1.1]); ax4.set_ylim([-1.1,1.1])
ax4.tick_params(axis='both', which='major', labelsize=labelsize)

ax5 = fig.add_subplot(gs[2:4, 2:4])
l2, = ax5.plot(xd[:res],W_d[tsp[1],idx], 'kx', markersize=10)  
l1, = ax5.plot(x,W_exact[tsp[1],:], 'b-', linewidth = 3)       
l3, = ax5.plot(x,W_pred_d[tsp[1],:], 'c--', linewidth = 2)
l4, = ax5.plot(x,W_pred_p[tsp[1],:], 'r--', linewidth = 2)
ax5.set_xlabel('$x$', fontsize = titlesize); ax5.set_ylabel('$w(x,t)$', fontsize = titlesize)
errord = np.amax(W_pred_d[tsp[1],:]-W_exact[tsp[1],:]) / np.amax(W_exact) * 100
errorp = np.amax(W_pred_p[tsp[1],:]-W_exact[tsp[1],:]) / np.amax(W_exact) * 100
ax5.set_title(titletxt.format(float(t[tsp[1]]), errord, errorp), fontsize = titlesize, pad=10)
ax5.set_aspect(0.5)
ax5.set_xlim([-0.1,1.1]); ax5.set_ylim([-1.1,1.1])
ax5.legend(loc='upper center', bbox_to_anchor=(0.4, -0.4), ncol=2, frameon=False)
ax5.tick_params(axis='both', which='major', labelsize=labelsize)

ax6 = fig.add_subplot(gs[2:4, 4:6])
ax6.plot(xd[:res],W_d[tsp[2],idx], 'kx', markersize=10, label = 'Noisy')     
ax6.plot(x,W_exact[tsp[2],:], 'b-', linewidth = 3, label = 'Exact')        
ax6.plot(x,W_pred_d[tsp[2],:], 'c--', linewidth = 2, label = 'Data-driven')
ax6.plot(x,W_pred_p[tsp[2],:], 'r--', linewidth = 2, label = 'Physics-informed')
ax6.set_xlabel('$x$', fontsize = titlesize); ax6.set_ylabel('$w(x,t)$', fontsize = titlesize)
errord = np.amax(W_pred_d[tsp[2],:]-W_exact[tsp[2],:]) / np.amax(W_exact) * 100
errorp = np.amax(W_pred_p[tsp[2],:]-W_exact[tsp[2],:]) / np.amax(W_exact) * 100
ax6.set_title(titletxt.format(float(t[tsp[2]]), errord, errorp), fontsize = titlesize, pad=10)
ax6.set_aspect(0.5)
ax6.set_xlim([-0.1,1.1]); ax6.set_ylim([-1.1,1.1])
ax6.tick_params(axis='both', which='major', labelsize=labelsize)

plt.tight_layout(0.8)
fig.legend([l1, l2, l3, l4], ['Exact Solution', 'Noisy Measurement', 'Data-driven', 'Physics-informed'], ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.008), frameon=False, fontsize = titlesize)

try:
    savefig('SSBt ' + str(layers[1:-1]))
except:
    pass 

try:
    savefig('./figures/SSBt ' + str(layers[1:-1]))
except:
    pass 