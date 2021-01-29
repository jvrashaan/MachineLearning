import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

class logisticRegression():
    #Given a data set of home features and home prices this class will predict the price of homes
    # given the features home size and number of bedrooms.
    def __init__(self):
        with open('ex2data1.txt') as f:
            data = f.read()
        
        self.data = np.array(re.split(',|\n', data)).reshape((100, 3)).astype(np.float)

    def sigmoid(self, z):
        #returns a matrix or array where every value is "activated" with the sigmoid function
        rows, cols = z.shape

        g = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                g[i, j] = 1/(1 + math.exp(-z[i, j]))
        
        return g
    
    def costFunction(self, theta, X, Y):
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
            grad[i, 0] = temp[i, 0] / m
        
        return cost, grad