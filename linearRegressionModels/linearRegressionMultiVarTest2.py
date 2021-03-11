import re
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from linearRegressionMultiVar import linear_regression_multi_var
from sklearn.preprocessing import StandardScaler

test = linear_regression_multi_var()
#get cost and grad 
theta = np.ones((2, 1))
X_1 = np.hstack((np.ones((len(test.X),1)), test.X))
cost, grad = test.compute_cost_multi_reg(X_1, test.Y, theta, 1)
#print('Cost and Grad at theta = [1 ; 1]: ', cost, grad)
theta, J_history = test.gradient_decent_multi_reg(X_1, test.Y, np.zeros((2,1)), 0.001, 4000, 0)

#plot cost history
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

#plot the data and linear fit
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.scatter(test.X, test.Y, marker="x", color="r")
x_value=[x for x in range(-50,40)]
y_value=[y * theta[1]+ theta[0] for y in x_value]
plt.plot(x_value, y_value, color="b")
plt.ylim(-5,40)
plt.xlim(-50,40)
plt.show()

#obtain learning curve and plot it
Xval_1 = np.hstack((np.ones((21,1)), test.Xval))
error_train, error_val = test.learning_curve(X_1, test.Y, Xval_1, test.Yval, 0)

plt.plot(range(12), error_train, label="Train")
plt.plot(range(12), error_val, label="Cross Validation", color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()

# Map X onto Polynomial features and normalize
p = 8
X_poly = test.poly_features(test.X, p)

scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)
X_poly = np.hstack((np.ones((X_poly.shape[0],1)), X_poly))

# Map Xtest onto polynomial features and normalize
X_poly_test = test.poly_features(test.Xtest, p)
X_poly_test = scaler.transform(X_poly_test)
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0],1)), X_poly_test))

# Map Xval onto polynomial features and normalize
X_poly_val = test.poly_features(test.Xval, p)
X_poly_val = scaler.transform(X_poly_val)
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0],1)), X_poly_val))

#minimize theta 
theta_poly, J_history_poly = test.gradient_decent_multi_reg(X_poly, test.Y, np.zeros((9,1)), 0.3, 20000, 0)

#plot poly fit
plt.scatter(test.X, test.Y , marker="x", color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value= np.linspace(-55,65,2400)

# Map the X values and normalize
x_value_poly = test.poly_features(x_value[:, np.newaxis], p)
x_value_poly = scaler.transform(x_value_poly)
x_value_poly = np.hstack((np.ones((x_value_poly.shape[0],1)), x_value_poly))
y_value = x_value_poly @ theta_poly
plt.plot(x_value, y_value, "--", color="b")
plt.show()

#plot learning curve
error_train, error_val = test.learning_curve(X_poly, test.Y, X_poly_val, test.Yval, 0)
plt.plot(range(12), error_train, label="Train")
plt.plot(range(12), error_val, label="Cross Validation", color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()

#poly regression with lamba = 100
theta_poly, J_history_poly = test.gradient_decent_multi_reg(X_poly, test.Y, np.zeros((9,1)), 0.01, 20000, 100)

plt.scatter(test.X, test.Y, marker="x", color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value = np.linspace(-55,65,2400)

# Map the X values and normalize
x_value_poly = test.poly_features(x_value[:, np.newaxis], p)
x_value_poly = scaler.transform(x_value_poly)
x_value_poly = np.hstack((np.ones((x_value_poly.shape[0],1)), x_value_poly))
y_value = x_value_poly @ theta_poly
plt.plot(x_value, y_value, "--", color="b")
plt.show()

#learning curve
error_train, error_val = test.learning_curve(X_poly, test.Y, X_poly_val, test.Yval, 100)
plt.plot(range(12), error_train, label="Train")
plt.plot(range(12), error_val, label="Cross Validation", color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()