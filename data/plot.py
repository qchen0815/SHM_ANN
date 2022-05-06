import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def display_fig(x, t, z):
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    """
    surf = ax.plot_surface(x, t, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    """
    #ax.contourf(t.T,x.T,z.T)
    ax.imshow(z.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[0.0, 10.0, 0.0, 1.0], 
                  origin='lower', aspect='auto')
    plt.show()

def load_data():
    data = np.loadtxt('data.txt').reshape([300, 300, 3])
    return data[:,:,0], data[:,:,1], data[:,:,2]

if __name__ == "__main__":
    xx, tt, uu = load_data()
    display_fig(xx, tt, uu)