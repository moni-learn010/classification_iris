import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#Load the iris dataset from sklearn
df = load_iris()
#print (df.feature_names)
#print (df.target_names)

df = pd.DataFrame(data=np.c_[df['data'],df['target']], columns=df['feature_names']+['target'])
x = df.drop(columns=['target'])
y = df['target']

#Train the KNN model
clf_knn = KNeighborsRegressor(n_neighbors=3)
clf_knn.fit(x, y)
# Save the model in pickle file
pkl_knn = "iris_knn.pkl"
with open(pkl_knn, 'wb') as file:
    pickle.dump(clf_knn, file)

#Train the model RandomForest
clf_random = RandomForestClassifier(n_estimators=100)
clf_random.fit(x, y)
# Save the model in pickle file
pkl_random = "iris_random.pkl"
with open(pkl_random, 'wb') as file:
    pickle.dump(clf_random, file)


#Train the model SVM
clf_svm = SVC(kernel='linear')
clf_svm.fit(x, y)
# Save the model in pickle file
pkl_svm = "iris_svm.pkl"
with open(pkl_svm, 'wb') as file:
    pickle.dump(clf_svm, file)

#Train the model logistic
clf_logistic = LogisticRegression(solver='lbfgs',max_iter=1000)
clf_logistic.fit(x, y)
# Save the model in pickle file
pkl_logistic = "iris_logistic.pkl"
with open(pkl_logistic, 'wb') as file:
    pickle.dump(clf_logistic, file)