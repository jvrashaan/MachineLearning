import numpy as np 
from supportVectorMachines import support_vector_machines
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import pandas as pd

test = support_vector_machines()

#plot data
m, n = test.X.shape[0], test.X.shape[1]
pos, neg = (test.Y == 1).reshape(m,1), (test.Y == 0).reshape(m,1)
plt.scatter(test.X[pos[:, 0], 0], test.X[pos[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(test.X[neg[:, 0], 0], test.X[neg[:, 0], 1], c="y", marker="o", s=50)
plt.show()

#init SVM classifier and fit linear boundary
classifier = SVC(kernel="linear")
classifier.fit(test.X, np.ravel(test.Y))

plt.scatter(test.X[pos[:, 0], 0], test.X[pos[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(test.X[neg[:, 0], 0], test.X[neg[:, 0], 1], c="y", marker="o", s=50)

# plotting the decision boundary
X_1, X_2 = np.meshgrid(np.linspace(test.X[:, 0].min(), test.X[:, 1].max(), num=100), np.linspace(test.X[:, 1].min(), test.X[:,1].max(), num=100))
plt.contour(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape), 1, colors="b")
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)
plt.show()

#C == 100
classifier2 = SVC(C=100, kernel="linear")
classifier2.fit(test.X, np.ravel(test.Y))

plt.figure(figsize=(8, 6))
plt.scatter(test.X[pos[:, 0], 0], test.X[pos[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(test.X[neg[:, 0], 0], test.X[neg[:, 0], 1], c="y", marker="o", s=50)

# plotting the decision boundary
X_3, X_4 = np.meshgrid(np.linspace(test.X[:, 0].min(), test.X[:, 1].max(), num=100), np.linspace(test.X[:, 1].min(), test.X[:, 1].max(), num=100))
plt.contour(X_3, X_4, classifier2.predict(np.array([X_3.ravel(), X_4.ravel()]).T).reshape(X_3.shape), 1, colors="b")
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)
plt.show()

#Init Guassian Kernels
m2, n2 = test.X2.shape[0], test.X2.shape[1]
pos2, neg2 = (test.Y2 == 1).reshape(m2, 1), (test.Y2 == 0).reshape(m2, 1)
plt.figure(figsize=(8, 6))
plt.scatter(test.X2[pos2[:, 0], 0], test.X2[pos2[:, 0], 1], c="r", marker="+")
plt.scatter(test.X2[neg2[:, 0], 0], test.X2[neg2[:, 0], 1], c="y", marker="o")
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.show()

classifier3 = SVC(kernel="rbf", gamma=30)
classifier3.fit(test.X2, test.Y2.ravel())

plt.scatter(test.X2[pos2[:, 0], 0], test.X2[pos2[:, 0], 1], c="r", marker="+")
plt.scatter(test.X2[neg2[:, 0], 0], test.X2[neg2[:, 0], 1], c="y", marker="o")

# plot the decision boundary
X_5, X_6 = np.meshgrid(np.linspace(test.X2[:, 0].min(), test.X2[:, 1].max(), num=100), np.linspace(test.X2[:, 1].min(), test.X2[:, 1].max(), num=100))
plt.contour(X_5, X_6, classifier3.predict(np.array([X_5.ravel(), X_6.ravel()]).T).reshape(X_5.shape), 1, colors="b")
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.show()

#init dataset 3
m3, n3 = test.X3.shape[0], test.X3.shape[1]
pos3, neg3= (test.Y3 == 1).reshape(m3, 1), (test.Y3 == 0).reshape(m3, 1)
plt.figure(figsize=(8, 6))
plt.scatter(test.X3[pos3[:, 0], 0], test.X3[pos3[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(test.X3[neg3[:, 0], 0], test.X3[neg3[:, 0], 1], c="y", marker="o", s=50)
plt.show()

#hyperparameter tuning
vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C, gamma = test.dataset_3_params(test.X3, test.Y3.ravel(), test.Xval, test.Yval.ravel(), vals)
classifier4 = SVC(C = C, gamma = gamma)
classifier4.fit(test.X3, test.Y3.ravel())

plt.figure(figsize=(8, 6))
plt.scatter(test.X3[pos3[:, 0], 0], test.X3[pos3[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(test.X3[neg3[:, 0], 0], test.X3[neg3[:, 0], 1], c="y", marker="o", s=50)

# plotting the decision boundary
X_7, X_8 = np.meshgrid(np.linspace(test.X3[:, 0].min(), test.X3[:, 1].max(), num=100), np.linspace(test.X3[:, 1].min(), test.X3[:, 1].max(), num=100))
plt.contour(X_7, X_8, classifier4.predict(np.array([X_7.ravel(), X_8.ravel()]).T).reshape(X_7.shape), 1, colors="b")
plt.xlim(-0.6, 0.3)
plt.ylim(-0.7, 0.5)
plt.show()

#spam classification
file_contents = open("emailSample1.txt", "r").read()
vocabList = open("vocab.txt", "r").read()

vocabList = vocabList.split("\n")[:-1]
#create key map
vocabList_d={}
for ea in vocabList:
    value,key = ea.split("\t")[:]
    vocabList_d[key] = value

#training svm for spam classification
word_indices = test.process_email(file_contents, vocabList_d)
features = test.email_features(word_indices, vocabList_d)
print("Length of feature vector: ", len(features))
print("Number of non-zero entries: ", np.sum(features))

spam_svc = SVC(C = 0.1, kernel = "linear")
spam_svc.fit(test.X_train, test.Y_train.ravel())
print("Training Accuracy:", (spam_svc.score(test.X_train, test.Y_train.ravel())) * 100, "%")

spam_svc.predict(test.X_test)
print("Test Accuracy:", (spam_svc.score(test.X_test, test.Y_test.ravel())) * 100, "%")

#top predictors for spam
weights = spam_svc.coef_[0]
weights_col = np.hstack((np.arange(1, 1900).reshape(1899, 1), weights.reshape(1899, 1)))
df = pd.DataFrame(weights_col)

df.sort_values(by=[1], ascending = False, inplace=True)

predictors = []
idx=[]
for i in df[0][:15]:
    for keys, values in vocabList_d.items():
        if str(int(i)) == values:
            predictors.append(keys)
            idx.append(int(values))

print("Top predictors of spam:")

for _ in range(15):
    print(predictors[_], "\t\t", round(df[1][idx[_]-1], 6))