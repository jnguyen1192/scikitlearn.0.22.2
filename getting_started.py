from sklearn.ensemble import RandomForestClassifier

print("FITTING AND PREDICITING: ESTIMATOR BASICS")
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
print("X (training dataset with 2 sample and 3 features", X)
y = [0, 1]
print("y (prediction values of each sample", y)

print("Training phase (normally ouput the training object)", clf.fit(X, y))
# Here we use a random forest classifier with a dataset X of 2 sample and 3 features each
# The prediction values for each sample are associate to y
# Now we are going to predict things

print("Prediction using X array (normally y will be predicted)", clf.predict(X))
print("Prediction using a new array (it will predict new things corresponding to samples and using training object", clf.predict([[4, 5, 6], [14, 15, 16]]))

