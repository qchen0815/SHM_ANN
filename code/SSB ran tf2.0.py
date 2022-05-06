"""
@author: Qiuyi Chen
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
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


#np.random.seed(1234)
#tf.random_set_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, exact_w, exact_w_t, N, layers, lb, ub):

        self.exact_w = exact_w
        self.exact_w_t = exact_w_t
        self.lb = lb
        self.ub = ub

        self.N = N
        
        # Initialize NNs
        self.layers = layers
        self.w = self.neural_net(layers)        
        
        # Optimizers
        self.optimizer_Adam = tf.keras.optimizers.Adam(0.00001)
        self.optimizer_lbfgs = tfp.optimizer.lbfgs_minimize
    
    def neural_net(self, layers):
        X = Input([None, layers[0]])
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in layers[1:-1]:
            H = Dense(l, activation='tanh')(H)
        Y = Dense(layers[-1], activation=None)(H)
        model = Model(inputs=X, outputs=Y, name='w')

        return model
    
    def net_w(self, x, t):
        with tf.GradientTape() as g3:
            g3.watch(x)
            with tf.GradientTape() as g2:
                g2.watch(x)
                with tf.GradientTape(persistent=True) as g1:
                    g1.watch([x,t])
                    X = tf.concat([x,t],1)
                    w = self.w(X)
                w_x = g1.gradient(w, x)
                w_t = g1.gradient(w, t)
            w_xx = g2.gradient(w_x, x)
        w_xxx = g3.gradient(w_xx, x)

        return w, w_t, w_x, w_xx, w_xxx

    def f_w(self, x, t):
        with tf.GradientTape(persistent=True) as g:
            g.watch([x, t])
            _, w_t, _, _, w_xxx = self.net_w(x,t)
        w_tt = g.gradient(w_t, t)
        w_xxxx = g.gradient(w_xxx, x)
        f_w = w_tt + w_xxxx

        return f_w

    def loss(self, x0, t0, w0, w_t0, x_lb, t_lb, x_ub, t_ub, x_f, t_f):
        w0_pred, w_t0_pred, _, _, _ = self.net_w(x0, t0) # initial points prediction
        w_lb_pred, _, _, w_xx_lb_pred, _ = self.net_w(x_lb, t_lb) # lb points prediction
        w_ub_pred, _, _, w_xx_ub_pred, _ = self.net_w(x_ub, t_ub) # ub points prediction
        f_w_pred = self.f_w(x_f, t_f) # governing equation

        mse = lambda res: tf.reduce_mean(tf.square(res))
        loss = mse(w0 - w0_pred) + mse(w_t0 - w_t0_pred) + \
            mse(w_lb_pred) + mse(w_xx_lb_pred) + mse(w_ub_pred) + mse(w_xx_ub_pred) + \
            mse(f_w_pred)

        return loss
    
    @tf.function
    def adam_train_step(self, inp):
        with tf.GradientTape() as t:
            current_loss = self.loss(*inp)
        dl_dW = t.gradient(current_loss, self.w.trainable_variables)
        self.optimizer_Adam.apply_gradients(zip(dl_dW, self.w.trainable_variables))

        return current_loss
    
    def lbfgs_train_step(self, inp):
        def val_grad(x):
            with tf.GradientTape() as t:
                current_loss = self.loss(*inp)
                dl_dW = t.gradient(current_loss, self.w.trainable_variables)
            return current_loss, dl_dW
        start = self.w.trainable_variables
        self.optimizer_lbfgs(
            val_grad,
            start,
            max_iterations=50000
        )
        
    
    def rand_points(self, n_0, n_b):
        x0 = np.random.uniform(lb[0], ub[0], [n_0, 1]); t0 = np.zeros([n_0, 1])
        w0 = self.exact_w(x0, 0); w_t0 = self.exact_w_t(x0, 0)

        x_lb = self.lb[0] * np.ones([n_b, 1]); t_lb = np.random.uniform(lb[1], ub[1], [n_b, 1])
        x_ub = self.ub[0] * np.ones([n_b, 1]); t_ub = np.random.uniform(lb[1], ub[1], [n_b, 1])

        X_f = self.lb + (self.ub-self.lb) * lhs(2, n_0 * n_b)
        x_f = X_f[:,0:1]; t_f = X_f[:,1:2]
        
        input_list = [x0, t0, w0, w_t0, x_lb, t_lb, x_ub, t_ub, x_f, t_f]

        return [tf.cast(each, tf.float32) for each in input_list]
    
    def grid_points(self, n_x=150, n_t=150):
        x = np.linspace(lb[0], ub[0], n_x)[:,None]
        t = np.linspace(lb[1], ub[1], n_t)[:,None]
        
        x0 = x; t0 = np.zeros(x0.shape)
        w0 = self.exact_w(x0, 0); w_t0 = self.exact_w_t(x0, 0)

        t_lb = t; x_lb = self.lb[0] * np.ones(t_lb.shape)
        t_ub = t; x_ub = self.ub[0] * np.ones(t_ub.shape)

        X, T = np.meshgrid(x, t)
        x_f = X.flatten()[:,None]; t_f = T.flatten()[:,None]
        
        input_list = [x0, t0, w0, w_t0, x_lb, t_lb, x_ub, t_ub, x_f, t_f]

        return [tf.cast(each, tf.float32) for each in input_list]

    def train(self, nIter):
        start_time = time.time()

        for it in range(nIter):
            inp = self.rand_points(self.N[0], self.N[1])
            loss_value = self.adam_train_step(inp)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        inp = self.grid_points()
        self.lbfgs_train_step(inp)
                                
    def predict(self, X_star):
        X_star = tf.cast(X_star, tf.float32)
    
        w_star = self.w(X_star).numpy()
        f_w_star = self.f_w(X_star[:,0:1], X_star[:,1:2]).numpy()
               
        return w_star, f_w_star

    
if __name__ == "__main__": 
    
    noise = 0.0        
    
    # Doman bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0]) 

    N_0 = 50 # number of initial points
    N_b = 50 # number of boundary points 
    N_f = 20000 # number of collocation points
    layers = [2, 100, 100, 100, 1]

    Exact_w = lambda x, t: np.sin(np.pi*x) * np.cos(np.pi**2*t)
    Exact_w_t = lambda x, t: - 2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi**2*t)
       
    ############################# Training ###############################

    model = PhysicsInformedNN(Exact_w, Exact_w_t, [N_0, N_b], layers, lb, ub) 
             
    start_time = time.time()                
    model.train(100)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    ############################ Predicting ##############################
    
    x = np.linspace(lb[0], ub[0], 250)[:,None]
    t = np.linspace(lb[1], ub[1], 250)[:,None]
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    w_pred, f_w_pred = model.predict(X_star)
    w_exact = Exact_w(X.flatten()[:,None], T.flatten()[:,None])
    d_w = w_pred - w_exact
    
    W_pred = griddata(X_star, w_pred.flatten(), (X, T), method='cubic')
    W_exact = griddata(X_star, w_exact.flatten(), (X, T), method='cubic')
    FW_pred = griddata(X_star, f_w_pred.flatten(), (X, T), method='cubic')
    D_W = griddata(X_star, d_w.flatten(), (X, T), method='cubic')
    
    ############################# Plotting ###############################
    fig = plt.figure(figsize=[5.4, 8.3], constrained_layout=True)
    gs = gridspec.GridSpec(4, 3, figure=fig)

    ####### Row 0: w(t,x) ##################
    ax = fig.add_subplot(gs[0, :])

    w = ax.imshow(W_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(w, cax=cax)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[0]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[200]*np.ones((2,1)), line, 'k--', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$w(t,x)$', fontsize = 10)

    ####### Row 1: dW(t,x) ##################
    ax = fig.add_subplot(gs[1, :])

    w = ax.imshow(D_W.T, interpolation='nearest', cmap='seismic',
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto', norm=DivergingNorm(0.))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(w, cax=cax)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(r'$\Delta w(t,x)$', fontsize = 10)

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
    ax.set_title('$f_w (t,x)$', fontsize = 10)

    ####### Row 3: w(t,x) slices ################## 
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(x,W_exact[0,:], 'b-', linewidth = 2, label = 'Exact')       
    ax4.plot(x,W_pred[0,:], 'r--', linewidth = 2, label = 'Prediction')
    ax4.set_xlabel('$x$'); ax4.set_ylabel('$w(t,x)$')    
    ax4.set_title('$t = %.2f$' % (t[0]), fontsize = 10)
    ax4.set_aspect(0.5)
    ax4.set_xlim([-0.1,1.1]); ax4.set_ylim([-1.1,1.1])
    

    ax5 = fig.add_subplot(gs[3, 1])
    l1, = ax5.plot(x,W_exact[100,:], 'b-', linewidth = 2)         
    l2, = ax5.plot(x,W_pred[100,:], 'r--', linewidth = 2)
    ax5.set_xlabel('$x$'); ax5.set_ylabel('$w(t,x)$')
    ax5.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax5.set_aspect(0.5)
    ax5.set_xlim([-0.1,1.1]); ax5.set_ylim([-1.1,1.1])  
    #ax5.legend(loc='upper center', bbox_to_anchor=(0.4, -0.4), ncol=2, frameon=False)

    ax6 = fig.add_subplot(gs[3, 2])
    ax6.plot(x,W_exact[200,:], 'b-', linewidth = 2, label = 'Exact')           
    ax6.plot(x,W_pred[200,:], 'r--', linewidth = 2, label = 'Prediction')
    ax6.set_xlabel('$x$'); ax6.set_ylabel('$w(t,x)$')
    ax6.set_title('$t = %.2f$' % (t[200]), fontsize = 10)
    ax6.set_aspect(0.5)
    ax6.set_xlim([-0.1,1.1]); ax6.set_ylim([-1.1,1.1])

    fig.legend([l1, l2], ['Exact', 'Prediction'], ncol=2, loc='lower center', bbox_to_anchor=(0.55, -0.012), frameon=False)
    
    try:
        savefig('SSB ' + str(layers[1:-1]))
    except:
        pass 

    try:
        savefig('./figures/SSB ' + str(layers[1:-1]))
    except:
        pass 
