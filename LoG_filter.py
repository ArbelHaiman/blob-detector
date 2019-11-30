# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:37:03 2015

@author: Amir
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def log_filt(ksize, sig):
    std2 = float( sig**2 )
    x = np.arange( -(ksize-1)/2, (ksize-1)/2+1, 1)
    y = np.arange( -(ksize-1)/2, (ksize-1)/2+1, 1)
    X, Y = np.meshgrid(x, y)

    arg = -(X*X + Y*Y)/(2*std2);

    h = np.exp(arg);

    eps = sys.float_info.epsilon
    h[h < eps*np.max(h)] = 0;

    sumh = np.sum(h)
    if sumh != 0:
        h = h/sumh

       # now calculate Laplacian
    h1 = h*(X*X + Y*Y - 2*std2)/(std2**2);
    h = h1 - np.sum(h1)/(ksize*ksize) # make the filter sum to zero
  
    return h

if __name__ == "__main__":
    sigma = 2
    k = 2**(0.25)
    filt_size =  2*np.ceil(3*sigma)+1 # filter size

    H = log_filt( filt_size, sigma);
    
    plt.imshow(H,interpolation='nearest')
    
    # 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange( -(filt_size-1)/2, (filt_size-1)/2+1, 1)
    Y = np.arange( -(filt_size-1)/2, (filt_size-1)/2+1, 1)
    X, Y = np.meshgrid(X, Y)
    
    surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='jet')
    ax.set_zlim(H.min(), H.max())

