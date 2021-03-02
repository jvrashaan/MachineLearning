import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from numpy import sin, cos, pi
from scipy.optimize import leastsq

class logistic_regression():
    #Given a data set of test scores for students this class will predict the probability that student gets
    #admitted.
    def __init__(self):
        with open('ex2data1.txt') as f:
            data = f.read()
        with open('ex2data2.txt') as f:
            data2 = f.read()
        self.data = np.array(re.split(',|\n', data)).reshape((100, 3)).astype(np.float)
        self.data2 = np.array(re.split(',|\n', data2)).reshape((118, 3)).astype(np.float)

    def sigmoid(self, z):
        #returns a matrix or array where every value is "activated" with the sigmoid function
        rows, cols = z.shape
        g = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                g[i, j] = (1)/(1 + np.exp(-z[i, j]))
        
        return g
    
    def cost_function(self, theta, X, Y):
        #Compute cost and gradient for logistic regression
        m = len(Y); # number of training examples
        f = len(X[0])
        cost = 0
        grad = np.zeros((theta.size, 1))

        for i in range(m):
            cost += -Y[i] * math.log(self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :])).reshape(1, 1))) - (1 - Y[i]) * np.log(1 - self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :])).reshape(1, 1)))

        cost = cost / m

        temp = np.zeros((f, 1))

        for i in range(m):
            for j in range(f):
                
                temp[j, 0] += ((self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :])).reshape(1, 1)) - Y[i]) * X[i, j])
            
        for i in range(f):
            grad[i, 0] = round(temp[i, 0] / m, 4)
        
        return cost, grad

    def predict(self, theta, X):
        #predict admission probability

        m = len(X)
        f = len(X[0])
        p = 0
        for i in range(m):
            if self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :]).reshape((f, 1)))) >= 0.5:
                p += 1
        
        return p / m

    def cost_function_reg(self, theta, X, Y, lamb):
        #Compute regularized cost and gradient for logistic regression
        m = len(Y); # number of training examples
        f = len(X[0])
        cost = 0
        grad = np.zeros((theta.size, 1))

        for i in range(m):
            cost += -Y[i] * math.log(self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :])).reshape(1, 1))) - (1 - Y[i]) * np.log(1 - self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :])).reshape(1, 1)))

        reg = 0
        for i in range(1, f):
            reg += theta[i, 0] ** 2
        
        reg *= lamb / (2 * m)
        cost = cost / m + reg

        temp = np.zeros((f, 1))

        for i in range(m):
            for j in range(f):
                
                temp[j, 0] += ((self.sigmoid(np.matmul(np.transpose(theta), np.transpose(X[i, :])).reshape(1, 1)) - Y[i]) * X[i, j])
            
        for i in range(f):
            if i > 1:
                grad[i, 0] = round((temp[i, 0] / m) + (lamb / m) * theta[i, 0], 4)
            else:
                grad[i, 0] = round(temp[i, 0] / m, 4)

        return cost, grad
    
    def map_feature(self, f1, f2):
        #maps the two input features
        #to quadratic features used in the regularization exercise.
        #f1 and f2 must be the same size
        degree = 6
        out = np.ones((len(f1), len(f1[0])))
        for i in range(1, degree + 1):
            for j in range(i + 1):
                out = np.hstack((out, np.around(np.multiply(np.power(f1, i-j), np.power(f2, j)), 6)))
        
        return out
    
    def find_boundary(self, x, y, n, plot_pts=1000):

        def sines(theta):
            ans = np.array([sin(i*theta)  for i in range(n+1)])
            return ans

        def cosines(theta):
            ans = np.array([cos(i*theta)  for i in range(n+1)])
            return ans

        def residual(params, x, y):
            x0 = params[0]
            y0 = params[1]
            c = params[2:]
            r_pts = ((x-x0)**2 + (y-y0)**2)**0.5

            thetas = np.arctan2((y-y0), (x-x0))
            m = np.vstack((sines(thetas), cosines(thetas))).T
            r_bound = m.dot(c)

            delta = r_pts - r_bound
            delta[delta>0] *= 10

            return delta

        # initial guess for x0 and y0
        x0 = np.mean(x)
        y0 = np.mean(y)
        params = np.zeros(2 + 2 * (n+1))

        params[0] = x0
        params[1] = y0
        params[2:] += 1000
        popt, pcov = leastsq(residual, x0=params, args=(x, y),
                            ftol=1.e-12, xtol=1.e-12)
        thetas = np.linspace(0, 2*pi, plot_pts)
        m = np.vstack((sines(thetas), cosines(thetas))).T
        c = np.array(popt[2:])
        r_bound = m.dot(c)
        x_bound = popt[0] + r_bound*cos(thetas)
        y_bound = popt[1] + r_bound*sin(thetas)

        return x_bound, y_bound