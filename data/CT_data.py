import numpy as np
from numpy import cos, sin, cosh, sinh
from random import uniform
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

w = np.array([1.875, 4.694, 7.855, 10.996, 14.137])

def cal_ui(x, t, w):
    ui = cos(w**2 * t) * (cosh(w*x) - cos(w*x) - \
        (cosh(w) + cos(w)) / (sinh(w) + sin(w)) * \
                (sinh(w*x) - sin(w*x)))
    return ui

def cal_u(x, t, w):
    xx, tt, ww = np.meshgrid(x, t, w, sparse=True)
    u = cal_ui(xx, tt, ww)
    rand = np.random.uniform(-1, 1, len(w)) / w
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
    t = np.linspace(0, 1, 300)
    u = cal_u(x, t, w[:3])
    print(u.shape)
    
    save_data(x, t, u)
    display_fig(x, t, u)