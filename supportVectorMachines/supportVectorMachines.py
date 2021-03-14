import numpy as np 
from scipy.io import loadmat
from sklearn.svm import SVC
from nltk.stem import PorterStemmer
import re

class support_vector_machines():
    #
    #
    def __init__(self):
        #init labeled data sets
        mat1 = loadmat("ex6data1.mat")
        mat2 = loadmat("ex6data2.mat")
        mat3 = loadmat("ex6data3.mat")
        spam_mat = loadmat("spamTrain.mat")
        spam_mat_test = loadmat("spamTest.mat")

        self.X = mat1["X"]
        self.Y = mat1["y"]
        self.X2 = mat2["X"]
        self.Y2 = mat2["y"]
        self.X3 = mat3["X"]
        self.Y3 = mat3["y"]
        self.Xval = mat3["Xval"]
        self.Yval = mat3["yval"]
        self.X_train = spam_mat["X"]
        self.Y_train = spam_mat["y"]
        self.X_test = spam_mat_test["Xtest"]
        self.Y_test = spam_mat_test["ytest"]

    def dataset_3_params(self, X, Y, Xval, Yval, vals):
        #Returns your choice of C and sigma.
        temp = 0
        best_c = 0
        best_gamma = 0
        for i in vals:
            C = i
            for j in vals:
                gamma = 1 / j
                classifier = SVC(C=C, gamma=gamma)
                classifier.fit(X, Y)
                prediction = classifier.predict(Xval)
                score = classifier.score(Xval, Yval)
                if score > temp:
                    temp = score
                    best_c = C
                    best_gamma = gamma

        return best_c, best_gamma

    def process_email(self, email_contents, vocabList_d):
        #Preprocesses the body of an email and returns a list of indices of the words contained in the email. 
        # Lower case
        email_contents = email_contents.lower()
        
        # Handle numbers
        email_contents = re.sub("[0-9]+", "number", email_contents)
        
        # Handle URLS
        email_contents = re.sub("[http|https]://[^\s]*", "httpaddr", email_contents)
        
        # Handle Email Addresses
        email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)
        
        # Handle $ sign
        email_contents = re.sub("[$]+", "dollar", email_contents)
        
        # Strip all special characters
        specialChar = ["<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":"]
        for char in specialChar:
            email_contents = email_contents.replace(str(char), "")
        email_contents = email_contents.replace("\n", " ")    
        
        # Stem the word
        ps = PorterStemmer()
        email_contents = [ps.stem(token) for token in email_contents.split(" ")]
        email_contents= " ".join(email_contents)
        
        # Process the email and return word_indices
        
        word_indices=[]
        
        for char in email_contents.split():
            if len(char) > 1 and char in vocabList_d:
                word_indices.append(int(vocabList_d[char]))
        
        return word_indices

    def email_features(self, word_indices, vocabList_d):
        #Takes in a word_indices vector and  produces a feature vector from the word indices. 
        
        n = len(vocabList_d)
        
        features = np.zeros((n, 1))
        
        for i in word_indices:
            features[i] = 1
            
        return features
