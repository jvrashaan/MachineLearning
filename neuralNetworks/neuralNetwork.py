import numpy as np
from numpy import sin, cos, pi
from scipy.io import loadmat

class neural_network():
    #Given a data set of test scores for students this class will predict the probability that student gets
    #admitted.
    def __init__(self):
        #init labeled data sets
        matrix = loadmat("ex3data1.mat")
        
        self.data = np.array(matrix["X"]).reshape((5000, 400)).astype(np.float)
        self.labels = np.array(matrix["y"]).reshape((5000, 1)).astype(np.float)

        matrix2 = loadmat("ex3weights.mat")

        self.theta1 = matrix2["Theta1"]
        self.theta2 = matrix2["Theta2"]

        matrix3 = loadmat("ex4data1.mat")
        
        self.data2 = np.array(matrix3["X"]).reshape((5000, 400)).astype(np.float)
        self.labels2 = np.array(matrix3["y"]).reshape((5000, 1)).astype(np.float)

        matrix4 = loadmat("ex4weights.mat")

        self.theta3 = matrix4["Theta1"]
        self.theta4 = matrix4["Theta2"]

    def sigmoid(self, z):
        #returns a matrix or array where every value is "activated" with the sigmoid function
        return (1)/(1 + np.exp(-z))
    
    def sigmoid_gradient(self, z):
        #computes the gradient of the sigmoid function
        sigmoid = self.sigmoid(z)
        return sigmoid *(1-sigmoid)
    
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
    
    def cost_function_ff(self, nn_params ,input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda):
        #computes cost function and gradient for nueral network using forward and backward propogation

        # Reshape nn_params back into the parameters Theta1 and Theta2
        Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
        Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

        m = X.shape[0]
        J=0
        X = np.hstack((np.ones((m,1)),X))
        y = np.zeros((m,num_labels))
        
        a1 = self.sigmoid(X @ Theta1.T)
        a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
        a2 = self.sigmoid(a1 @ Theta2.T) # output layer
        
        for i in range(1,num_labels+1):
            y[:,i-1][:,np.newaxis] = np.where(Y==i,1,0)
        for j in range(num_labels):
            J = J + sum(-y[:,j] * np.log(a2[:,j]) - (1-y[:,j])*np.log(1-a2[:,j]))
        
        cost = 1/m* J
        reg_J = cost + Lambda/(2*m) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))

        # Implement the backpropagation algorithm to compute the gradients
    
        grad1 = np.zeros((Theta1.shape))
        grad2 = np.zeros((Theta2.shape))
        
        for i in range(m):
            xi= X[i,:] # 1 X 401
            a1i = a1[i,:] # 1 X 26
            a2i =a2[i,:] # 1 X 10
            d2 = a2i - y[i,:]
            d1 = Theta2.T @ d2.T * self.sigmoid_gradient(np.hstack((1,xi @ Theta1.T)))
            grad1= grad1 + d1[1:][:,np.newaxis] @ xi[:,np.newaxis].T
            grad2 = grad2 + d2.T[:,np.newaxis] @ a1i[:,np.newaxis].T
            
        grad1 = 1/m * grad1
        grad2 = 1/m*grad2
        
        grad1_reg = grad1 + (Lambda/m) * np.hstack((np.zeros((Theta1.shape[0],1)),Theta1[:,1:]))
        grad2_reg = grad2 + (Lambda/m) * np.hstack((np.zeros((Theta2.shape[0],1)),Theta2[:,1:]))
        
        return cost, grad1, grad2,reg_J, grad1_reg,grad2_reg


    def gradientDescent(self, theta, X, Y, Lambda, alpha, num_iters):
        #Compute gradient for neural network 
        J_history =[]
        
        for i in range(num_iters):
            cost, grad = self.cost_function(theta, X, Y, Lambda)
            theta = theta - (alpha * grad)
            J_history.append(cost)

        return theta , J_history
    
    def gradient_descent_ff(self, X, Y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size, hidden_layer_size, num_labels):
        #Take in numpy array X, Y and theta and update theta by taking num_iters gradient steps
        #with learning rate of alpha
        #return theta and the list of the cost of theta during each iteration
        
        Theta1 = initial_nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
        Theta2 = initial_nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
        
        m=len(Y)
        J_history =[]
        
        for i in range(num_iters):
            nn_params = np.append(Theta1.flatten(),Theta2.flatten())
            cost, grad1, grad2 = self.cost_function_ff(nn_params,input_layer_size, hidden_layer_size, num_labels, X, Y, Lambda)[3:]
            Theta1 = Theta1 - (alpha * grad1)
            Theta2 = Theta2 - (alpha * grad2)
            J_history.append(cost)
        
        nn_paramsFinal = np.append(Theta1.flatten(),Theta2.flatten())
        return nn_paramsFinal , J_history
    
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
    
    def randInitializeWeights(self, L_in, L_out):
        #randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
        
        epi = (6**1/2) / (L_in + L_out)**1/2
        
        W = np.random.rand(L_out,L_in +1) *(2*epi) -epi
        
        return W