import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class anomaly_detection():
    def __init__(self):
        mat = loadmat("ex8data1.mat")
        self.X = mat["X"]
        self.Xval = mat["Xval"]
        self.Yval = mat["yval"]

        mat2 = loadmat("ex8data2.mat")
        self.X2 = mat2["X"]
        self.Xval2 = mat2["Xval"]
        self.Yval2 = mat2["yval"]

        mat3 = loadmat("ex8_movies.mat")
        mat4 = loadmat("ex8_movieParams.mat")
        self.Y = mat3["Y"] # 1682 X 943 matrix, containing ratings (1-5) of 1682 movies on 943 user
        self.R = mat3["R"] # 1682 X 943 matrix, where R(i,j) = 1 if and only if user j give rating to movie i
        self.X = mat4["X"] # 1682 X 10 matrix , num_movies X num_features matrix of movie features
        self.Theta = mat4["Theta"] # 943 X 10 matrix, num_users X num_features matrix of user features

    def estimate_gaussian(self, X):
        #This function estimates the parameters of a Gaussian distribution using the data in X
        
        m = X.shape[0]
        #compute mean
        sum_ = np.sum(X, axis=0)
        mu = (1 / m) * sum_
        
        # compute variance
        var = (1 / m) * np.sum((X - mu) ** 2, axis=0)
        
        return mu, var

    def multi_variate_gaussian(self, X, mu, sigma2):
        #Computes the probability density function of the multivariate gaussian distribution.
        
        k = len(mu)
        
        sigma2 = np.diag(sigma2)
        X = X - mu.transpose()
        p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma2) ** 0.5)) * np.exp(-0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
        return p
    
    def select_threshold(self, yval, pval):
        #Find the best threshold (epsilon) to use for selecting outliers
        
        best_epi = 0
        best_F1 = 0
        
        stepsize = (max(pval) - min(pval))/1000
        epi_range = np.arange(pval.min(), pval.max(), stepsize)
        for epi in epi_range:
            predictions = (pval < epi)[:, np.newaxis]
            tp = np.sum(predictions[yval == 1] == 1)
            fp = np.sum(predictions[yval == 0] == 1)
            fn = np.sum(predictions[yval == 1] == 0)
            
            # compute precision, recall and F1
            prec = tp/(tp + fp)
            rec = tp/(tp + fn)
            
            F1 = (2* prec * rec)/(prec + rec)
            
            if F1 > best_F1:
                best_F1 = F1
                best_epi = epi
            
        return best_epi, best_F1

    def cofi_cost_func(self, params, Y, R, num_users, num_movies, num_features, Lambda):
        #Returns the cost and gradient for the collaborative filtering problem
        
        # Unfold the params
        X = params[:num_movies * num_features].reshape(num_movies, num_features)
        Theta = params[num_movies * num_features:].reshape(num_users, num_features)
        
        predictions =  X @ Theta.transpose()
        err = (predictions - Y)
        J = 1/2 * np.sum((err ** 2) * R)
        
        #compute regularized cost function
        reg_X =  Lambda/2 * np.sum(Theta ** 2)
        reg_Theta = Lambda/2 * np.sum(X ** 2)
        reg_J = J + reg_X + reg_Theta
        
        # Compute gradient
        X_grad = err * R @ Theta
        Theta_grad = (err* R).transpose() @ X
        grad = np.append(X_grad.flatten(), Theta_grad.flatten())
        
        # Compute regularized gradient
        reg_X_grad = X_grad + Lambda * X
        reg_Theta_grad = Theta_grad + Lambda * Theta
        reg_grad = np.append(reg_X_grad.flatten(), reg_Theta_grad.flatten())
        
        return J, grad, reg_J, reg_grad

    def normalize_ratings(self, Y, R):
        #normalized Y so that each movie has a rating of 0 on average, and returns the mean rating in Ymean.
        
        m, n = Y.shape[0], Y.shape[1]
        Ymean = np.zeros((m, 1))
        Ynorm = np.zeros((m, n))
        
        for i in range(m):
            Ymean[i] = np.sum(Y[i, :])/ np.count_nonzero(R[i, :])
            Ynorm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Ymean[i]
            
        return Ynorm, Ymean

    def gradient_descent(self, initial_parameters, Y, R, num_users, num_movies, num_features, alpha, num_iters, Lambda):
        #Optimize X and Theta
        # unfold the parameters
        X = initial_parameters[:num_movies * num_features].reshape(num_movies, num_features)
        Theta = initial_parameters[num_movies * num_features:].reshape(num_users, num_features)
        
        J_history =[]
        
        for i in range(num_iters):
            params = np.append(X.flatten(), Theta.flatten())
            cost, grad = self.cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda)[2:]
            
            # unfold grad
            X_grad = grad[:num_movies * num_features].reshape(num_movies, num_features)
            Theta_grad = grad[num_movies * num_features:].reshape(num_users, num_features)
            X = X - (alpha * X_grad)
            Theta = Theta - (alpha * Theta_grad)
            J_history.append(cost)
        
        paramsFinal = np.append(X.flatten(), Theta.flatten())
        return paramsFinal , J_history