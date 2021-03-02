import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from numpy import sin, cos, pi
from scipy.optimize import leastsq, minimize, fmin_cg, fmin_bfgs
from scipy.io import loadmat

class neural_network():
    #Given a data set of test scores for students this class will predict the probability that student gets
    #admitted.
    def __init__(self):
        matrix = loadmat("ex3data1.mat")
        
        self.data = np.array(matrix["X"]).reshape((5000, 400)).astype(np.float)
        self.labels = np.array(matrix["y"]).reshape((5000, 1)).astype(np.float)

        matrix2 = loadmat("ex3weights.mat")

        self.theta1 = matrix2["Theta1"]
        self.theta2 = matrix2["Theta2"]

    def sigmoid(self, z):
        #returns a matrix or array where every value is "activated" with the sigmoid function
        return (1)/(1 + np.exp(-z))
    
    def cost_function(self, theta, X, Y, Lambda):
        #Compute cost for neural network 
        m = len(Y)
        #compute cost
        predictions = self.sigmoid(X @ theta)
        error = (-Y * np.log(predictions)) - ((1.0-Y)* np.log(1.0-predictions))
        cost = 1/m * sum(error)
        regCost = cost + Lambda/(2 * m) * sum(theta[1:]**2)
        
        # compute gradient
        j_0= 1/m * (X.transpose() @ (predictions - Y))[0]
        j_1 = 1/m * (X.transpose() @ (predictions - Y))[1:] + (Lambda/m)* theta[1:]
        grad= np.vstack((j_0[:,np.newaxis],j_1))
        return regCost[0], grad
    
    def gradientDescent(self, theta, X, Y, Lambda, alpha, num_iters):
        #Compute gradient for neural network 
        J_history =[]
        
        for i in range(num_iters):
            cost, grad = self.cost_function(theta, X, Y, Lambda)
            theta = theta - (alpha * grad)
            J_history.append(cost)

        return theta , J_history
    
    def one_vs_all(self, num_labels, X, Y, Lambda):
        #computes one vs all classification
        m = len(X)
        n = len(X[0])
        all_theta = []
        X = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((n + 1, 1))
        all_J = []
        for c in range(1, num_labels + 1):
            theta, J_history = self.gradientDescent(theta, X, np.where(Y==c,1,0), Lambda, 1, 300)
            all_theta.extend(theta)
            all_J.extend(J_history)
        
        return np.array(all_theta).reshape(num_labels, n+1), all_J
    
    def predict_one_vs_all(self, all_theta, X):
        #Using all_theta, compute the probability of X(i) for each class and predict the label
        #return a vector of prediction
        m = X.shape[0]
        X = np.hstack((np.ones((m,1)),X))
        predictions = X @ all_theta.transpose()
        return np.argmax(predictions, axis=1) + 1
    
    def predict_fp(self, theta1, theta2, X):
        #Predict the label of an input given a trained neural network
        
        m = X.shape[0]
        X = np.hstack((np.ones((m,1)),X))
        
        a1 = self.sigmoid(X @ theta1.transpose())
        a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
        a2 = self.sigmoid(a1 @ theta2.transpose()) # output layer
        
        return np.argmax(a2, axis=1) + 1
    