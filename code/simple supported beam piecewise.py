"""
@author: Qiuyi Chen
"""

import tensorflow as tf
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


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, w_0, w_t0, layers, lb, ub):

        self.w0 = w0
        self.w_t0 = w_t0
        
        self.lb = lb
        self.ub = ub
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.w0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.w_t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # tf Graphs
        self.w0_pred, self.w_t0_pred, _, _, _ = self.net_w(self.x0_tf, self.t0_tf) # initial points prediction
        self.w_lb_pred, self.w_tlb_pred, _, self.w_xx_lb_pred, _ = self.net_w(self.x_lb_tf, self.t_lb_tf) # lb points prediction
        self.w_ub_pred, self.w_tub_pred, _, self.w_xx_ub_pred, _ = self.net_w(self.x_ub_tf, self.t_ub_tf) # ub points prediction
        self.f_w_pred = self.net_f_w(self.x_f_tf, self.t_f_tf) # governing equation
        
        # Loss

        self.loss = tf.reduce_mean(tf.square(self.w0_tf - self.w0_pred)) + \
                    tf.reduce_mean(tf.square(self.w_t0_tf - self.w_t0_pred)) + \
                    tf.reduce_mean(tf.square(self.w_lb_pred)) + \
                    tf.reduce_mean(tf.square(self.w_tlb_pred)) + \
                    tf.reduce_mean(tf.square(self.w_xx_lb_pred)) + \
                    tf.reduce_mean(tf.square(self.w_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.w_tub_pred)) + \
                    tf.reduce_mean(tf.square(self.w_xx_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_w_pred))

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer(0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        #self.W_1 = tf.Variable(tf.ones([1,2], dtype=tf.float32))
        #self.b_1 = tf.Variable(tf.zeros([1,2], dtype=tf.float32))
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            if l > 0:
                H = H + tf.tanh(tf.add(tf.matmul(H, W), b))
            else:
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_w(self, x, t):
        X = tf.concat([x,t],1)
        
        w = self.neural_net(X, self.weights, self.biases)

        w_t = tf.gradients(w, t)[0]

        w_x = tf.gradients(w, x)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_xxx = tf.gradients(w_xx, x)[0]

        return w, w_t, w_x, w_xx, w_xxx

    def net_f_w(self, x, t):
        _, w_t, _, _, w_xxx = self.net_w(x,t)

        w_tt = tf.gradients(w_t, t)[0]

        w_xxxx = tf.gradients(w_xxx, x)[0]
        
        f_w = w_tt + w_xxxx

        return f_w
    
    def rand_points(self, n_0, n_b):
        x0 = np.linspace(self.lb[0], self.ub[0], 150)[:,None]; t0 = np.zeros(x0.shape)
        w0 = self.w0; w_t0 = self.w_t0

        x_lb = self.lb[0] * np.ones([n_b, 1]); t_lb = np.random.uniform(self.lb[1], self.ub[1], [n_b, 1])
        x_ub = self.ub[0] * np.ones([n_b, 1]); t_ub = np.random.uniform(self.lb[1], self.ub[1], [n_b, 1])

        X_f = self.lb + (self.ub-self.lb) * lhs(2, n_0 * n_b)
        x_f = X_f[:,0:1]; t_f = X_f[:,1:2]

        tf_dict = {
            self.x0_tf: x0, self.t0_tf: t0,
            self.w0_tf: w0, self.w_t0_tf: w_t0,
            self.x_lb_tf: x_lb, self.t_lb_tf: t_lb,
            self.x_ub_tf: x_ub, self.t_ub_tf: t_ub,
            self.x_f_tf: x_f, self.t_f_tf: t_f
            }

        return tf_dict

    
    def grid_points(self, n_x=150, n_t=150):
        x = np.linspace(self.lb[0], self.ub[0], n_x)[:,None]
        t = np.linspace(self.lb[1], self.ub[1], n_t)[:,None]
        
        x0 = x; t0 = np.zeros(x0.shape)
        w0 = self.w0; w_t0 = self.w_t0

        t_lb = t; x_lb = self.lb[0] * np.ones(t_lb.shape)
        t_ub = t; x_ub = self.ub[0] * np.ones(t_ub.shape)

        X, T = np.meshgrid(x, t)
        x_f = X.flatten()[:,None]; t_f = T.flatten()[:,None]

        tf_dict = {
            self.x0_tf: x0, self.t0_tf: t0,
            self.w0_tf: w0, self.w_t0_tf: w_t0,
            self.x_lb_tf: x_lb, self.t_lb_tf: t_lb,
            self.x_ub_tf: x_ub, self.t_ub_tf: t_ub,
            self.x_f_tf: x_f, self.t_f_tf: t_f
            }

        return tf_dict
    
    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):
        start_time = time.time()

        for it in range(nIter):
            tf_dict = self.rand_points(50, 50)
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        tf_dict = self.grid_points()                                                                                      
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
             
                                    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        w_star = self.sess.run(self.w0_pred, tf_dict)
        w_t_star = self.sess.run(self.w_t0_pred, tf_dict)
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]} 
        
        f_w_star = self.sess.run(self.f_w_pred, tf_dict)
               
        return w_star, f_w_star, w_t_star

def stitch_data(data_list):
    l = np.concatenate([data[:-1] for data in data_list])
    stitched_data = np.concatenate([l, data_list[-1][-1:]])
    return stitched_data
    
if __name__ == "__main__":      
    steps = 20

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 10.0])

    Exact_w = lambda x, t: np.sin(np.pi*x) * np.cos(np.pi**2*t)
    Exact_w_t = lambda x, t: - np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi**2*t)

    x0 = np.linspace(lb[0], ub[0], 150)[:,None]; t0 = np.zeros(x0.shape)
    w0 = Exact_w(x0, 0); w_t0 = Exact_w_t(x0, 0)

    W_pred_list = []; FW_pred_list = []; W_exact_list = []; D_W_list = []

    for n in range(steps):
        # Domain bounds
        layers = [2, 64, 1]
        
        ############################# Training ###############################

        model = PhysicsInformedNN(w0, w_t0, layers, lb, np.array([ub[0], ub[1]/steps])) 
                
        start_time = time.time()
        model.train(5000)
        elapsed = time.time() - start_time 
        print('Training time: %.4f' % (elapsed))

        ############################ Predicting ##############################
        
        x = np.linspace(lb[0], ub[0], 150)[:,None]
        t = np.linspace(lb[1], ub[1]/steps, 25)[:,None]
        X, T = np.meshgrid(x,t)
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        w_pred, f_w_pred, w_t_pred = model.predict(X_star)
        w_exact = Exact_w(X.flatten()[:,None], T.flatten()[:,None]+ub[1]/steps*n)
        d_w = w_pred - w_exact
        
        W_pred_list.append(griddata(X_star, w_pred.flatten(), (X, T), method='cubic'))
        W_exact_list.append(griddata(X_star, w_exact.flatten(), (X, T), method='cubic'))
        FW_pred_list.append(griddata(X_star, f_w_pred.flatten(), (X, T), method='cubic'))
        D_W_list.append(griddata(X_star, d_w.flatten(), (X, T), method='cubic'))

        # reset ics
        w0 = w_pred[-150:]; w_t0 = w_t_pred[-150:]


    W_pred = stitch_data(W_pred_list)
    W_exact = stitch_data(W_exact_list)
    FW_pred = stitch_data(FW_pred_list)
    D_W = stitch_data(D_W_list)
    t = np.linspace(lb[1], ub[1], len(W_pred))[:,None]

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
    ax.plot(t[198]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[393]*np.ones((2,1)), line, 'k--', linewidth = 1)
    
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
    l1, = ax5.plot(x,W_exact[198,:], 'b-', linewidth = 2)         
    l2, = ax5.plot(x,W_pred[198,:], 'r--', linewidth = 2)
    ax5.set_xlabel('$x$'); ax5.set_ylabel('$w(t,x)$')
    ax5.set_title('$t = %.2f$' % (t[198]), fontsize = 10)
    ax5.set_aspect(0.5)
    ax5.set_xlim([-0.1,1.1]); ax5.set_ylim([-1.1,1.1])  
    #ax5.legend(loc='upper center', bbox_to_anchor=(0.4, -0.4), ncol=2, frameon=False)

    ax6 = fig.add_subplot(gs[3, 2])
    ax6.plot(x,W_exact[393,:], 'b-', linewidth = 2, label = 'Exact')           
    ax6.plot(x,W_pred[393,:], 'r--', linewidth = 2, label = 'Prediction')
    ax6.set_xlabel('$x$'); ax6.set_ylabel('$w(t,x)$')
    ax6.set_title('$t = %.2f$' % (t[393]), fontsize = 10)
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
