import re, struct
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from neuralNetwork import neural_network
from PIL import Image
import matplotlib.image as mpimg

test = neural_network()
#initialize small test data
theta_t = np.array([-2, -1, 1, 2]).reshape((4, 1))
X_t = np.hstack((np.ones((5,1)), np.divide(np.transpose(np.array(range(1, 16)).reshape((3, 5))), 10)))
y_t = np.array([1,0,1,0,1]).reshape((5, 1))
lambda_t = 3

#calculate cost and gradient
cost, grad = test.cost_function(theta_t, X_t, y_t, lambda_t)
#print("Cost: ", cost)
#print("Gradient: ", grad)

#visualize pixel data 
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(test.data[np.random.randint(0,5001),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off")

#train network
num_labels = 10
lamb = 0.1
all_theta, all_J = test.one_vs_all(num_labels, test.data, test.labels, lamb)

#plot costs over iterations
plt.plot(all_J[0:50])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
#plt.show()

#Predict labels for all data using training network
pred = test.predict_one_vs_all(all_theta, test.data)
#print("Training Set Accuracy:",sum(pred[:,np.newaxis]==test.labels)[0]/5000*100,"%")

#predict labels for data using feed forward propogation
pred2 = test.predict_fp(test.theta1, test.theta2, test.data)
#print("Training Set Accuracy:",sum(pred2[:,np.newaxis]==test.labels)[0]/5000*100,"%")

#Neural network using feed forward propogation and back propogation with initial theta values
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
nn_params = np.append(test.theta3.flatten(), test.theta4.flatten())
J,reg_J = test.cost_function_ff(nn_params, input_layer_size, hidden_layer_size, num_labels, test.data2, test.labels2, 1)[0:4:3]
print("Cost at parameters (non-regularized):",J,"\nCost at parameters (Regularized):",reg_J)

#Need to break symmetry so randomize weights
initial_Theta1 = test.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = test.randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(),initial_Theta2.flatten())

#train network and predict labels
nnTheta, nnJ_history = test.gradient_descent_ff(test.data2, test.labels2, initial_nn_params, 0.8, 800, 1, input_layer_size, hidden_layer_size, num_labels)
Theta1 = nnTheta[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size+1)
Theta2 = nnTheta[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels, hidden_layer_size+1)
pred3 = test.predict(Theta1, Theta2, test.data2)
print("Training Set Accuracy:",sum(pred3[:,np.newaxis]==test.labels2)[0]/5000*100,"%")