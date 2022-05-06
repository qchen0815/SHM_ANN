import numpy as np
from numpy import cos, sin, cosh, sinh
from random import uniform
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

w = np.arange(1, 2)

def cal_ui(x, t, w):
    ui = np.sin(w*np.pi*x) * np.cos(2*(w*np.pi)**2*t)
    return ui

def cal_u(x, t, w):
    xx, tt, ww = np.meshgrid(x, t, w, sparse=True)
    u = cal_ui(xx, tt, ww)
    rand = 1 #np.random.uniform(-1, 1, len(w)) #/ w
    return np.sum(u * rand, axis=2)

def display_fig(x, t, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, T = np.meshgrid(x, t)
    surf = ax.plot_surface(T, X, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

def save_data(x, t, u):
    X, T = np.meshgrid(x, t)
    X = X.flatten()[:,None]
    T = T.flatten()[:,None]
    U = u.flatten()[:,None]
    data = np.concatenate([X, T, U], axis=1)
    np.savetxt('data.txt', data)

if __name__ == "__main__":
    x = np.linspace(0, 1, 300)
    t = np.linspace(0, 0.3, 300)
    u = cal_u(x, t, w[:])
    print(u.shape)
    
    save_data(x, t, u)
    display_fig(x, t, u)