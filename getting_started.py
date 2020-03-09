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

print("PIPELINES: CHAINNING PRE-PROCESSORS AND ESTIMATORS")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
print("Pipeline:\n", pipe)
X, y = load_iris(return_X_y=True)
print("Iris dataset X:\n", X[:5])
print("Iris predictions y:\n", y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("Separate the dataset into trainset and testset...")
print("Iris dataset train X_train:\n", X_train[:5])
print("Iris predictions train y_train:\n", y_train[:5])
print("Iris dataset test X_test:\n", X_test[:5])
print("Iris predictions test y_test:\n", y_test[:5])
print("Fit the pipeline:\n", pipe.fit(X_train, y_train))
print("Get the score using the testset:\n", accuracy_score(pipe.predict(X_test), y_test))
print("-" * 400)

print("MODEL EVALUATION")
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
X, y = make_regression(n_samples=1000, random_state=0)
print("Create dataset X for a regression:\n", X[:5][:5])
print("Create prediction y for a regression:\n", y[:5])
lr = LinearRegression()
print("Create Linear Regression classifier\n", lr)
result = cross_validate(lr, X, y)
print("Use cross validation on dataset X with prediction y using 5 fold CV\n:", result)
print("Get the test score:\n", result['test_score'])
print("-" * 400)

print("AUTOMATIC PARAMETER SEARCHES")
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
print("California housing dataset X:\n", X[:5])
print("California housing predictions y:\n", y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("Separate the dataset into trainset and testset...")
print("Iris dataset train X_train:\n", X_train[:5])
print("Iris predictions train y_train:\n", y_train[:5])
print("Iris dataset test X_test:\n", X_test[:5])
print("Iris predictions test y_test:\n", y_test[:5])
param_distributions = {'n_estimators': randint(1, 5),
                       'max_depth': randint(5, 10)}
print("Create dict with param to find with intervals\n", param_distributions)
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                            n_iter=5,
                            param_distributions=param_distributions,
                            random_state=0)
print("Create the object to search the good parameters using a specify estimator:\n", search)
print("Search the parameters using the previous object and the training dataset:\n", search.fit(X_train, y_train))
print("Get the best param:\n", search.best_params_)
print("Use the model with the new search parameters that have been automatically associated:\n", search.score(X_test, y_test))
print("-" * 400)