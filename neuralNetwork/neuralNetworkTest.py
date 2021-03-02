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
print("Cost: ", cost)
print("Gradient: ", grad)

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
print("Training Set Accuracy:",sum(pred[:,np.newaxis]==test.labels)[0]/5000*100,"%")

#predict labels for data using feed forward propogation
pred2 = test.predict_fp(test.theta1, test.theta2, test.data)
print("Training Set Accuracy:",sum(pred2[:,np.newaxis]==test.labels)[0]/5000*100,"%")