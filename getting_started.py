print("FITTING AND PREDICITING: ESTIMATOR BASICS")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
print("X (training dataset with 2 sample and 3 features:\n", X)
y = [0, 1]
print("y (prediction values of each sample:\n", y)
print("Training phase (normally ouput the training object):\n", clf.fit(X, y))
# Here we use a random forest classifier with a dataset X of 2 sample and 3 features each
# The prediction values for each sample are associate to y
# Now we are going to predict things
print("Prediction using X array (normally y will be predicted):\n", clf.predict(X))
print("Prediction using a new array (it will predict new things corresponding to samples and using training object:\n", clf.predict([[4, 5, 6], [14, 15, 16]]))
print("-" * 400)

print("TRANSOFRMERS AND PRE-PROCESSORS")
from sklearn.preprocessing import StandardScaler
X = [[0 , 15],
     [1, -10]]
print("Dataset X with 2 features and 2 samples:\n", X)
print("Transform the dataset unsing a transformer object:\n", StandardScaler().fit(X).transform(X))
print("-" * 400)

