import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Helper function for permuting the rows of 2 arrays
def shuffle_arrs(a , b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

X = np.load('dataNewReshape.npy') # reshaped data from (19..., 21,3 ) to (19..., 63)
y = np.load('labelsNew.npy')
full_data = np.hstack((X,y.reshape(-1,1)))

just_vowels = full_data[np.isin(full_data[:,-1], [0, 4, 8, 14, 20])]
X_vow = just_vowels[:,:-1]
y_vow = just_vowels[:,-1]

models = {
    'Naive Bayes' : GaussianNB(),
    'SVM' : SVC(),
    'KNN' : KNeighborsClassifier(n_neighbors=6)
}

Xt, yt = shuffle_arrs(X, y)
Vt, vt = shuffle_arrs(X_vow, y_vow)

scores = []
vowel_scores = []
for m in models:
    print(m)
    model = models[m]
    score = cross_val_score(model, Xt, yt)
    vscore = cross_val_score(model, Vt, vt)
    scores.append(np.mean(score))
    vowel_scores.append(np.mean(vscore))

print(scores)
print(vowel_scores)
