from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
from scipy.linalg import lstsq

def lsqfit(xs, ys, yys, zs):
    A = np.stack([np.ones(len(xs)), xs, ys, yys])
    print(A)
    p, _, _, _ = lstsq(A.T, zs)
    return p

data = np.loadtxt('trainingresult.csv', delimiter=',', skiprows=1).T
num = 5

xs1 = data[1][:num]
ys1 = data[2][:num]
xxs1 = xs1**2
yys1 = ys1**2
zs1 = data[3][:num] * 100

xs2 = data[1][7:7+num]
ys2 = data[2][7:7+num]
xxs2 = xs2**2
yys2 = ys2**2
zs2 = data[3][7:7+num] * 100

p1 = lsqfit(xs1, ys1, yys1, zs1)
p2 = lsqfit(xs2, ys2, yys2, zs2)

xx, yy = np.meshgrid(np.linspace(10, 20, 100), np.linspace(5, 20, 100))
zz1 = p1[0] + p1[1]*xx + p1[2]*yy + p1[3]*yy**2
zz2 = p2[0] + p2[1]*xx + p2[2]*yy + p2[3]*yy**2
zz3 = np.ones(xx.shape) * 5 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs1, ys1, zs1, marker='o', s=100, label='physics informed')
ax.scatter(xs2, ys2, zs2, marker='^', s=100, label='data-driven')
ax.plot_surface(xx, yy, zz1, alpha=0.5)
ax.plot_surface(xx, yy, zz2, alpha=0.5)
#ax.plot_surface(xx, yy, zz3, alpha=0.2, color='red')
ax.legend()

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Number of sensors')
ax.zaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_zlabel('Maximum Error')

plt.show()

"""
def lsqfit(xs, ys, xxs, yys, zs):
    A = np.stack([np.ones(len(xs)), xs, ys, xxs, yys])
    print(A)
    p, _, _, _ = lstsq(A.T, zs)
    return p

data = np.loadtxt('trainingresult.csv', delimiter=',', skiprows=1).T
num = 6

xs1 = data[1][:num]
ys1 = data[2][:num]
xxs1 = xs1**2
yys1 = ys1**2
zs1 = data[3][:num] * 100

xs2 = data[1][num:]
ys2 = data[2][num:]
xxs2 = xs2**2
yys2 = ys2**2
zs2 = data[3][num:] * 100

p1 = lsqfit(xs1, ys1, xxs1, yys1, zs1)
p2 = lsqfit(xs2, ys2, xxs2, yys2, zs2)

xx, yy = np.meshgrid(np.linspace(10, 20, 100), np.linspace(5, 20, 100))
zz1 = p1[0] + p1[1]*xx + p1[2]*yy + p1[3]*xx**2 + p1[4]*yy**2
zz2 = p2[0] + p2[1]*xx + p2[2]*yy + p2[3]*xx**2 + p2[4]*yy**2
zz3 = np.ones(xx.shape) * 5 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs1, ys1, zs1, marker='o', s=100, label='physics informed')
ax.scatter(xs2, ys2, zs2, marker='^', s=100, label='data-driven')
ax.plot_surface(xx, yy, zz1, alpha=0.5)
ax.plot_surface(xx, yy, zz2, alpha=0.5)
#ax.plot_surface(xx, yy, zz3, alpha=0.2, color='red')
ax.legend()

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Number of sensors')
ax.zaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_zlabel('Maximum Error')

plt.show()
"""