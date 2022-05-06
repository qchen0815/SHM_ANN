
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import cos, sin, cosh, sinh

plt.rcParams.update({'font.size': 13})

steps = 6

lb = np.array([0.0, 0.0])
ub = np.array([1.0, 0.15])
layers = [2, 64, 64, 64, 64, 64, 64, 64, 64, 1]

Exact_w = lambda x, t: np.sin(3*np.pi*x) * np.cos(9*np.pi**2*t)
Exact_w_t = lambda x, t: - 9*np.pi**2 * np.sin(3*np.pi*x) * np.sin(9*np.pi**2*t)

x0 = np.linspace(lb[0], ub[0], 150)[:,None]; t0 = np.zeros(x0.shape)
w0 = Exact_w(x0, 0); w_t0 = Exact_w_t(x0, 0)

W_pred = np.load("piecewise/w_pred.npy")
W_exact = np.load("piecewise/w_exact.npy")
FW_pred = np.load("piecewise/fw_pred.npy")
D_W = np.load("piecewise/d_w.npy")
print(len(W_pred))

x = np.linspace(lb[0], ub[0], 250)[:,None]
t = np.linspace(lb[1], ub[1], len(W_pred))[:,None]

############################# Plotting ###############################
fig = plt.figure(figsize=[5.4, 6.5], constrained_layout=True)
gs = gridspec.GridSpec(3, 3, figure=fig)

####### Row 0: w(t,x) ##################
ax = fig.add_subplot(gs[0, :])

w = ax.imshow(W_exact.T, interpolation='nearest', cmap='YlGnBu', 
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(w, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[49]*np.ones((2,1)), line, 'b--', linewidth = 1)
ax.plot(t[147]*np.ones((2,1)), line, 'b--', linewidth = 1)
ax.plot(t[196]*np.ones((2,1)), line, 'b--', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$w(x,t)$', fontsize = 13)

####### Row 1: dW(t,x) ##################
ax = fig.add_subplot(gs[1, :])

w = ax.imshow(W_pred.T, interpolation='nearest', cmap='YlGnBu',
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', vmin=-np.amax(np.abs(W_exact)), vmax=np.amax(np.abs(W_exact)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(w, cax=cax)

ax.plot(t[49]*np.ones((2,1)), line, 'r--', linewidth = 1)
ax.plot(t[147]*np.ones((2,1)), line, 'r--', linewidth = 1)
ax.plot(t[196]*np.ones((2,1)), line, 'r--', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title(r'$\hat{w}(x,t)$', fontsize = 13)

"""
####### Row 2: FW(t,x) ##################
ax = fig.add_subplot(gs[2, :])

w = ax.imshow(FW_pred.T, interpolation='nearest', cmap='PuOr', 
                extent=[lb[1], ub[1], lb[0], ub[0]], 
                origin='lower', aspect='auto', norm=DivergingNorm(0.))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(w, cax=cax)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$f_w (x,t)$', fontsize = 13)
"""
####### Row 3: w(t,x) slices ################## 
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(x,W_exact[49,:], 'b-', linewidth = 2, label = 'Exact')       
ax4.plot(x,W_pred[49,:], 'r--', linewidth = 2, label = 'Prediction')
ax4.set_xlabel('$x$'); ax4.set_ylabel('$w(x,t)$')    
ax4.set_title('$t = %.3f$' % (t[49]), fontsize = 13)
ax4.set_aspect(0.5)
ax4.set_xlim([-0.1,1.1]); ax4.set_ylim([-1.1,1.1])


ax5 = fig.add_subplot(gs[2, 1])
l1, = ax5.plot(x,W_exact[147,:], 'b-', linewidth = 2)         
l2, = ax5.plot(x,W_pred[147,:], 'r--', linewidth = 2)
ax5.set_xlabel('$x$'); ax5.set_ylabel('$w(x,t)$')
ax5.set_title('$t = %.3f$' % (t[147]), fontsize = 13)
ax5.set_aspect(0.5)
ax5.set_xlim([-0.1,1.1]); ax5.set_ylim([-1.1,1.1])  
#ax5.legend(loc='upper center', bbox_to_anchor=(0.4, -0.4), ncol=2, frameon=False)

ax6 = fig.add_subplot(gs[2, 2])
ax6.plot(x,W_exact[196,:], 'b-', linewidth = 2, label = 'Exact Solution')           
ax6.plot(x,W_pred[196,:], 'r--', linewidth = 2, label = 'MLP Prediction')
ax6.set_xlabel('$x$'); ax6.set_ylabel('$w(x,t)$')
ax6.set_title('$t = %.3f$' % (t[196]), fontsize = 13)
ax6.set_aspect(0.5)
ax6.set_xlim([-0.1,1.1]); ax6.set_ylim([-1.1,1.1])

fig.legend([l1, l2], ['Exact Solution', 'MLP Prediction'], ncol=2, loc='lower center', bbox_to_anchor=(0.55, -0.012), frameon=False)

try:
    savefig('piece ' + str(layers[1:-1]))
except:
    pass 

try:
    savefig('./figures/piece ' + str(layers[1:-1]))
except:
    pass 