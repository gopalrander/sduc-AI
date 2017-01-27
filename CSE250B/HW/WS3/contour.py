# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:42:08 2016

@author: gopal
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
 
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
var = multivariate_normal(mean=[0,0], cov=[[8, 4],[4,8]])
plt.contourf(x, y, var.pdf(pos))
plt.show()


#def gaussian_2d(x, y, x0, y0, xsig, ysig):
#    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))
#
#delta = 0.025
#x = np.arange(-3.0, 3.0, delta)
#y = np.arange(-2.0, 2.0, delta)
#X, Y = np.meshgrid(x, y)
#Z1 = gaussian_2d(X, Y, 0., 0., 2., 2.)
## difference of Gaussians
#
## Create a contour plot with labels using default colors.  The
## inline argument to clabel will control whether the labels are draw
## over the line segments of the contour, removing the lines beneath
## the label
#plt.clf()
#CS = plt.contour(X, Y, Z1)
#plt.clabel(CS, inline=1, fontsize=10)
#plt.title('Simplest default with labels')    