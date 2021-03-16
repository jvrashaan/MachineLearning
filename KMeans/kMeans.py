import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import svd

class k_means():
    def __init__(self):
        mat = loadmat("ex7data2.mat")
        self.X = mat["X"]

        mat2 = loadmat("bird_small.mat")
        self.A = mat2["A"]

        mat3 = loadmat("ex7data1.mat")
        self.X3 = mat3["X"]

        mat4 = loadmat("ex7faces.mat")
        self.X4 = mat4["X"]

    def find_closest_centroids(self, X, centroids):
        #Returns the closest centroids in idx for a dataset X where each row is a single example.
        K = centroids.shape[0]
        idx = np.zeros((X.shape[0], 1))
        temp = np.zeros((centroids.shape[0], 1))
        
        for i in range(X.shape[0]):
            for j in range(K):
                dist = X[i, :] - centroids[j, :]
                length = np.sum(dist ** 2)
                temp[j] = length
            idx[i] = np.argmin(temp) + 1
        return idx

    def compute_centroids(self, X, idx, K):
        #returns the new centroids by computing the means of the data points assigned to each centroid.
        
        m, n = X.shape[0], X.shape[1]
        centroids = np.zeros((K, n))
        count = np.zeros((K, 1))
        
        for i in range(m):
            index = int((idx[i] - 1)[0])
            centroids[index, :] += X[i, :]
            count[index] += 1
        
        return centroids / count

    def plot_k_means(self, X, centroids, idx, K, num_iters):
        #plots the data points with colors assigned to each centroid
        m, n = X.shape[0], X.shape[1]
        
        fig, ax = plt.subplots(nrows=num_iters, ncols=1, figsize=(6, 36))
        
        for i in range(num_iters):    
            # Visualisation of data
            color = "rgb"
            for k in range(1,K+1):
                grp = (idx == k).reshape(m, 1)
                ax[i].scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k - 1], s=15)

            # visualize the new centroids
            ax[i].scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
            title = "Iteration Number " + str(i)
            ax[i].set_title(title)
            
            # Compute the centroids mean
            centroids = self.compute_centroids(X, idx, K)
            
            # assign each training example to the nearest centroid
            idx = self.find_closest_centroids(X, centroids)
        
        plt.tight_layout()
        plt.show()
    
    def k_means_init_centroids(self, X, K):
        #This function initializes K centroids randomly
        
        m, n = X.shape[0], X.shape[1]
        centroids = np.zeros((K, n))
        
        for i in range(K):
            centroids[i] = X[np.random.randint(0, m + 1), :]
            
        return centroids

    def run_k_means(self, X, initial_centroids, num_iters, K):
        idx = self.find_closest_centroids(X, initial_centroids)
        
        for i in range(num_iters):
            
            # Compute the centroids mean
            centroids = self.compute_centroids(X, idx, K)

            # assign each training example to the nearest centroid
            idx = self.find_closest_centroids(X, initial_centroids)

        return centroids, idx

    def feature_normalize(self, X):
        #Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
        
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        
        X_norm = (X - mu) / sigma
        
        return X_norm, mu, sigma
        
    def pca(self, X):
        #Computes eigenvectors of the covariance matrix of X
        
        m, n = X.shape[0], X.shape[1]
        
        sigma = 1 / m * X.transpose() @ X
        
        U, S, V = svd(sigma)
        
        return U, S, V

    def project_data(self, X, U, K):
        #Computes the reduced data representation when projecting only on to the top k eigenvectors
        
        m = X.shape[0]
        U_reduced = U[:, :K]
        Z = np.zeros((m, K))
        
        for i in range(m):
            for j in range(K):
                Z[i, j] = X[i, :] @ U_reduced[:, j]
        
        return Z

    def recover_data(self, Z, U, K):
        #Recovers an approximation of the original data when using the projected data
    
        m, n = Z.shape[0], U.shape[0]
        X_rec = np.zeros((m, n))
        U_reduced = U[:, :K]
        
        for i in range(m):
            X_rec[i, :] = Z[i, :] @ U_reduced.transpose()
        
        return X_rec