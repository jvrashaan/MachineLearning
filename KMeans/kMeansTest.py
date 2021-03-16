from kMeans import k_means
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 

test = k_means()
# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
#find closest centroids
idx = test.find_closest_centroids(test.X, initial_centroids)
print("Closest centroids for the first 3 examples:\n", idx[0:3])

#compute centroids given X
centroids = test.compute_centroids(test.X, idx, K)
print("Centroids computed after initial finding of closest centroids:\n", centroids)

#compute and plot k means
m, n = test.X.shape[0], test.X.shape[1]
test.plot_k_means(test.X, initial_centroids, idx, K, 10)

centroids = test.k_means_init_centroids(test.X, K)
idx = test.find_closest_centroids(test.X, centroids)
test.plot_k_means(test.X, centroids, idx, K, 10)

# preprocess and reshape the image
X2 = (test.A/255).reshape(128*128,3)

# Running K-means algorithm on the data
K2 = 16
num_iters = 10
initial_centroids2 = test.k_means_init_centroids(X2, K2)
centroids2, idx2 = test.run_k_means(X2, initial_centroids2, num_iters, K2)

m2, n2 = test.X.shape[0], test.X.shape[1]
X2_recovered = X2.copy()
for i in range(1, K2 + 1):
    X2_recovered[(idx2 == i).ravel(), :] = centroids2[i - 1]

# Reshape the recovered image into proper dimensions
X2_recovered = X2_recovered.reshape(128,128,3)

# Display the image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X2.reshape(128, 128, 3))
ax[1].imshow(X2_recovered)

plt.scatter(test.X3[:, 0], test.X3[:, 1], marker="o", facecolors="none", edgecolors="b")

#visualizing PCA
X_norm, mu, std = test.feature_normalize(test.X3)
U, S = test.pca(X_norm)[:2]

plt.scatter(test.X3[:, 0], test.X3[:, 1], marker="o", facecolors="none", edgecolors="b")
plt.plot([mu[0], (mu + 1.5 * S[0] * U[:, 0].transpose())[0]], [mu[1], (mu + 1.5 * S[0] * U[:, 0].transpose())[1]], color="black", linewidth=3)
plt.plot([mu[0], (mu + 1.5 * S[1] * U[:, 1].transpose())[0]], [mu[1], (mu + 1.5 * S[1] * U[:, 1].transpose())[1]], color="black", linewidth=3)
plt.xlim(-1,7)
plt.ylim(2,8)
plt.plot()

print("Top eigenvector U(:,1) =:", U[:, 0])

# Project the data onto K=1 dimension
K = 1
Z = test.project_data(X_norm, U, K)
print("Projection of the first example:", Z[0][0])

X_rec  = test.recover_data(Z, U, K)
print("Approximation of the first example:", X_rec[0, :])

#visualizing the projections
plt.scatter(X_norm[:, 0], X_norm[:, 1], marker="o", label="Original", facecolors="none", edgecolors="b", s=15)
plt.scatter(X_rec[:, 0], X_rec[:, 1], marker="o", label="Approximation", facecolors="none", edgecolors="r", s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()
plt.show()

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
for i in range(0, 100, 10):
    for j in range(10):
        ax[int(i/10), j].imshow(test.X4[i + j,:].reshape(32, 32, order="F"), cmap="gray")
        ax[int(i/10), j].axis("off")

plt.show()

X_norm2 = test.feature_normalize(test.X4)[0]

# Run PCA
U2 = test.pca(X_norm2)[0]

#Visualize the top 36 eigenvectors found
U_reduced = U2[:, :36].transpose()
fig2, ax2 = plt.subplots(6, 6, figsize=(8, 8))
for i in range(0,36,6):
    for j in range(6):
        ax2[int(i/6),j].imshow(U_reduced[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        ax2[int(i/6),j].axis("off")

plt.show()

K2 = 100
Z2 = test.project_data(X_norm2, U2, K2)
print("The projected data Z has a size of:", Z2.shape)

X_rec2 = test.recover_data(Z2, U2, K2)

# Visualize the reconstructed data
fig3, ax3 = plt.subplots(10, 10, figsize=(8, 8))
for i in range(0,100,10):
    for j in range(10):
        ax3[int(i/10),j].imshow(X_rec2[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        ax3[int(i/10),j].axis("off")

plt.show()