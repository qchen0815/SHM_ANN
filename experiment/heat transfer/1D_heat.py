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
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, w0, w_t0, tb, X_f, layers, lb, ub):
        """
        x0, u0, v0: points for initial condition and corresponding values
        tb: time points for boundary condition
        X_f: collocation points that have to satisfy the governing equation
        layers: structure of MLP
        lb: lower bound [x, t]
        ub: upper bound [x, t]
        """

        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0) initial points
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb) lower boundary points
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb) upper boundary points
        
        self.lb = lb
        self.ub = ub
               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.w0 = w0
        self.w_t0 = w_t0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.w0_tf = tf.placeholder(tf.float32, shape=[None, self.w0.shape[1]])
        self.w_t0_tf = tf.placeholder(tf.float32, shape=[None, self.w_t0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.w0_pred, self.w_t0_pred, _ = self.net_w(self.x0_tf, self.t0_tf) # initial points prediction
        self.w_lb_pred, _, self.w_x_lb_pred = self.net_w(self.x_lb_tf, self.t_lb_tf) # lb points prediction
        self.w_ub_pred, _, self.w_x_ub_pred = self.net_w(self.x_ub_tf, self.t_ub_tf) # ub points prediction
        self.f_w_pred = self.net_f_w(self.x_f_tf, self.t_f_tf) # governing equation
        
        # Loss
        # ordinary + boundary + boundary differentiation + gonverning equation
        self.loss = 100 * tf.reduce_mean(tf.square(self.w0_tf - self.w0_pred)) + \
                    tf.reduce_mean(tf.square(self.w_x_lb_pred)) + \
                    tf.reduce_mean(tf.square(self.w_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_w_pred))
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
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
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # normalization
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # no activation
        return Y
    
    def net_w(self, x, t):
        X = tf.concat([x,t],1)
        
        w = self.neural_net(X, self.weights, self.biases)

        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]

        return w, w_t, w_x

    def net_f_w(self, x, t):
        _, w_t, w_x = self.net_w(x,t)

        w_xx = tf.gradients(w_x, x)[0]

        c = 1
        
        f_w = c * w_xx - w_t
        
        return f_w
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.w0_tf: self.w0, self.w_t0_tf: self.w_t0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
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
               
        return w_star, w_t_star, f_w_star
    
if __name__ == "__main__": 
     
    noise = 0.0        
    
    # Doman bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 0.3])

    N0 = 50 # number of initial points
    N_b = 50 # number of boundary points 
    N_f = 20000 # number of collocation points
    layers = [2, 1000, 1]
    
    t = np.arange(lb[1], ub[1], (ub[1]-lb[1])/N_b/5)[:,None]
    x = np.arange(lb[0], ub[0], (ub[0]-lb[0])/N0/5)[:,None]
    Exact_w = np.cos(np.pi * x) + 1
    
    X, T = np.meshgrid(x,t) # points for plotting
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    w_star = Exact_w.T.flatten()[:,None]
    #v_star = Exact_v.T.flatten()[:,None]
    #h_star = Exact_h.T.flatten()[:,None]
    
    ###########################
    
    # Initial condition
    idx_x = np.random.choice(x.shape[0], N0 * 5, replace=False)
    x0 = x[idx_x,:]
    w0 = Exact_w[idx_x,0:1]
    
    # Boundary condition. Only symmetry is required, so no value is needed
    idx_t = np.random.choice(t.shape[0], N_b * 5, replace=False)
    tb = t[idx_t,:]
    
    X_f = lb + (ub-lb)*lhs(2, N_f) # generate collocation point indices using latin hypercube
            
    model = PhysicsInformedNN(x0, w0, w0, tb, X_f, layers, lb, ub) 
             
    start_time = time.time()                
    model.train(10000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
        
    w_pred, w_t_pred, f_w_pred = model.predict(X_star)
            
    #error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    #error_w_t = np.linalg.norm(w_t_star-w_t_pred,2)/np.linalg.norm(w_star,2)
    #print('Error w: %e' % (error_w))
    #print('Error w_t: %e' % (error_w_t))

    
    W_pred = griddata(X_star, w_pred.flatten(), (X, T), method='cubic')

    FW_pred = griddata(X_star, f_w_pred.flatten(), (X, T), method='cubic') 
    

    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(W_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[50]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[150]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[200]*np.ones((2,1)), line, 'k--', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$w(t,x)$', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-2/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    #ax.plot(x,Exact_w[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,W_pred[0,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$w(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[0]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,2.1])    
    
    ax = plt.subplot(gs1[0, 1])
    #ax.plot(x,Exact_w[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,W_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$w(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,2.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    #ax.plot(x,Exact_w[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,W_pred[200,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$w(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,2.1])    
    ax.set_title('$t = %.2f$' % (t[200]), fontsize = 10)
    
    try:
        savefig('NLS')
    except:
        pass 

    try:
        savefig('./figures/NLS')
    except:
        pass 
      