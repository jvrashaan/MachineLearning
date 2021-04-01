import numpy as np
import matplotlib.pyplot as plt
from anomalyDectection import anomaly_detection
import pandas as pd

test = anomaly_detection()
#plot Gaussian Distribution
plt.scatter(test.X[:, 0], test.X[:, 1], marker="x")
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.plot()

mu, sigma2 = test.estimate_gaussian(test.X)
#print(mu.shape, sigma2.shape)
#multi variate gaussian
p = test.multi_variate_gaussian(test.X, mu, sigma2)
#print(p)
#plot fit
plt.figure(figsize=(8, 6))
plt.scatter(test.X[:, 0], test.X[:, 1], marker="x")
X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
p2 = test.multi_variate_gaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
contour_level = 10 ** np.array([np.arange(-20, 0, 3, dtype=np.float)]).transpose()
plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()

#select threshold
pval = test.multi_variate_gaussian(test.Xval, mu, sigma2)
epsilon, F1 = test.select_threshold(test.Yval, pval)
print("Best epsilon found using cross-validation:", epsilon)
print("Best F1 on Cross Validation Set:",F1)

#visualize anomalies
plt.figure(figsize=(8, 6))

# plot the data
plt.scatter(test.X[:, 0],test.X[:,n1], marker="x")

# potting of contour
X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
p2 = test.multi_variate_gaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
contour_level = 10 ** np.array([np.arange(-20, 0, 3, dtype=np.float)]).transpose()
plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)

# Circling of anomalies
outliers = np.nonzero(p < epsilon)[0]
plt.scatter(test.X[outliers, 0], test.X[outliers, 1], marker ="o", facecolor="none", edgecolor="r", s=70)

plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()

#high dimensional data set
#compute the mean and variance
mu2, sigma2_2 = test.estimate_gaussian(test.X2)

# Training set
p3 = test.multi_variate_gaussian(test.X2, mu2, sigma2_2)

# cross-validation set
pval2 = test.multi_variate_gaussian(test.Xval2, mu2, sigma2_2)

# Find the best threshold
epsilon2, F1_2 = test.select_threshold(test.Yval2, pval2)
print("Best epsilon found using cross-validation:", epsilon2)
print("Best F1 on Cross Validation Set:", F1_2)
print("# Outliers found:", np.sum(p3 < epsilon2))

# Compute average rating 
print("Average rating for movie 1 (Toy Story):", np.sum(test.Y[0, :] * test.R[0, :])/np.sum(test.R[0,:]), "/5")

#plot the ratings matrix
plt.figure(figsize=(8, 16))
plt.imshow(test.Y)
plt.xlabel("Users")
plt.ylabel("Movies")
plt.show()

# Reduce the data set size to run faster
num_users, num_movies, num_features = 4, 5, 3
X_test = test.X[:num_movies, :num_features]
Theta_test= test.Theta[:num_users, :num_features]
Y_test = test.Y[:num_movies, :num_users]
R_test = test.R[:num_movies, :num_users]
params = np.append(X_test.flatten(), Theta_test.flatten())

# Evaluate cost function
J, grad = test.cofi_cost_func(params, Y_test, R_test, num_users, num_movies, num_features, 0)[:2]
print("Cost at loaded parameters:", J)

J2, grad2 = test.cofi_cost_func(params, Y_test, R_test, num_users, num_movies, num_features, 1.5)[2:]
print("Cost at loaded parameters (lambda = 1.5):", J2)

# load movie list
movieList = open("movie_ids.txt", "r").read().split("\n")[:-1]

# see movie list
np.set_printoptions(threshold = np.nan)
movieList

# Initialize my ratings
my_ratings = np.zeros((1682,1))

# Create own ratings
my_ratings[0] = 4 
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[82]= 4
my_ratings[225] = 5
my_ratings[354]= 5

print("New user ratings:\n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated", int(my_ratings[i]),"for index", movieList[i])

test.Y = np.hstack((my_ratings, test.Y))
test.R = np.hstack((my_ratings != 0, test.R))

# Normalize Ratings
Ynorm, Ymean = test.normalize_ratings(Y, R)

num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set initial Parameters (Theta,X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.append(X.flatten(), Theta.flatten())
Lambda = 10

# Optimize parameters using Gradient Descent
paramsFinal, J_history = test.gradient_descent(initial_parameters, Y, R, num_users, num_movies, num_features, 0.001, 400, Lambda)

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

# unfold paramaters
X = paramsFinal[:num_movies * num_features].reshape(num_movies, num_features)
Theta = paramsFinal[num_movies * num_features:].reshape(num_users, num_features)

# Predict rating
p = X @ Theta.transpose()
my_predictions = p[:, 0][:, np.newaxis] + Ymean

df = pd.DataFrame(np.hstack((my_predictions, np.array(movieList)[:, np.newaxis])))
df.sort_values(by=[0], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

print("Top recommendations for you:\n")
for i in range(10):
    print("Predicting rating",round(float(df[0][i]), 1)," for index", df[1][i])