import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.io import loadmat

class linear_regression_multi_var():
    #Given a data set of home features and home prices this class will predict the price of homes
    # given the features home size and number of bedrooms.
    def __init__(self):
        with open('ex1data2.txt') as f:
            data = f.read()
        
        self.data = np.array(re.split(',|\n', data)).reshape((47, 3)).astype(np.float)

        matrix = loadmat("ex5data1.mat")
        self.X = matrix["X"]
        self.Y = matrix["y"]
        self.Xtest = matrix["Xtest"]
        self.Ytest = matrix["ytest"]
        self.Xval = matrix["Xval"]
        self.Yval = matrix["yval"]

    def normalize_features(self, X):
        #returns a normalized version of X where
        #the mean value of each feature is 0 and the standard deviation
        #is 1.
        x_norm = X
        mu = np.zeros((1, len(X[0])))
        sigma = np.zeros((1, len(X[0])))
        features = len(X[0])
        for i in range(features):
            mu[0, i] = np.mean(x_norm, axis=0)[i]
            x_norm[:, i] = x_norm[:, i] - mu[0, i]
            sigma[0, i] = np.std(x_norm, axis=0)[i]
            x_norm[:, i] = x_norm[:, i] / sigma[0, i]
        
        return [x_norm, mu, sigma]

    def gradient_decent_multi(self, X, Y, alpha, theta, iterations):
        #Performs gradient descent to learn theta
        #updates theta by taking num_iters gradient steps with learning rate alpha
        #m = # of samples
        # does not use regulization
        m = len(Y)
        J_history = np.zeros((iterations, 1))
        features = len(X[0])
        for i in range(iterations):
            temp = np.zeros((features, 1))
            for j in range(m):
                for k in range(features):
                    temp[k, 0] += (np.matmul(np.transpose(theta), np.transpose(X[j, :].reshape((1, features)))) - Y[j])[0, 0] *  X[j, k]
        
            for j in range(features):
                theta[j, 0] -= alpha * (1 / m) * temp[j, 0]

            # Save the cost J in every iteration    
            J_history[i] = self.compute_cost_multi(X, Y, theta)
        
        return theta, J_history

    def compute_cost_multi(self, X, Y, theta):
        #Compute cost for linear regression with multiple variables
        #m = # of samples
        # does not use regulization
        m = len(Y)

        temp = 0
        for i in range(m):
            cost = np.matmul(np.transpose(theta), np.transpose(X[i, :]))[0]
            temp += (cost - Y[i]) ** 2
        
        return temp /(2 * m)

    def normal_equation(self, X, Y):
        #returns the calculated theta values using normal equations

        return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
    
    def compute_cost_multi_reg(self, X, Y, theta, Lambda):
        #regularized cost function compute cost and gradient

        m = len(X)
        n = len(X[0])
        cost = sum(np.power(((X @ theta) - Y), 2))/ (2 * m) + ((Lambda / (2 * m)) * sum(np.power(theta[1:], 2)))
        
        temp = np.zeros((n, 1))
        for i in range(m):
            for j in range(n):
                temp[j] = temp[j] + (((np.transpose(theta) @ X[i]) - Y[i]) * X[i, j])
        
        grad = np.zeros(theta.shape)
        for i in range(n):
            if i > 1:
                grad[i] = temp[i] / m + (Lambda/m * theta[i])
            else:
                grad[i] = temp[i] / m
        
        return cost, grad
    
    def gradient_decent_multi_reg(self, X, Y, theta, alpha, num_iters, Lambda):
        #computes optimal values for theta using regularized cost function

        m = len(Y)
        J_history =[]
        
        for i in range(num_iters):
            cost, grad = self.compute_cost_multi_reg(X, Y, theta, Lambda)
            theta = theta - (alpha * grad)
            J_history.append(cost)
    
        return theta , J_history

    def learning_curve(self, X, Y, Xval, Yval, Lambda=0):
        #Generates the train and cross validation set errors needed 
        #to plot a learning curve. Lambda is set to zero to obtain errors.
        m = len(X)
        n = len(X[0])
        error_train = []
        error_val = []

        for i in range(1,m+1):
            theta = self.gradient_decent_multi_reg(X[0:i,:], Y[0:i,:], np.zeros((n,1)), 0.001, 3000, Lambda)[0]
            error_train.append(self.compute_cost_multi_reg(X[0:i,:], Y[0:i,:], theta, Lambda)[0])
            error_val.append(self.compute_cost_multi_reg(Xval, Yval, theta, Lambda)[0])
            
        return error_train, error_val

    def poly_features(self, X, p):
        #Takes a data matrix X (size m x 1) and maps each example into its polynomial features where 
        #X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    
        for i in range(2, p + 1):
            X = np.hstack((X ,(X[:,0] ** i)[:,np.newaxis]))
        
        return X
        
    