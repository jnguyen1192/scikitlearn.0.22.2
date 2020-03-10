print("1.1 Linear Models")
print("Linear combination of the features with y as the predicted value\nThe formula is :\n" + " " * 15 + " y(w, x) = w0 + w1x1 + ... + wpxp\n" + " " * 20 + "with w the interception(difference) and x the dataset")
print("-"*500)
print("\t"*1 + "1.1.1 Ordinary Least Squares")
print("Solve the problem min(w)||Xw - y||²2")
from sklearn import linear_model
reg = linear_model.LinearRegression()
print("Create the linear regression object:\n", reg)
print("Train the linear regression object:\n", reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2]))
print("Get the coefficient of the linear regression:\n", reg.coef_)
print("This method cost O(n_samples * n²_features")
# TODO bonus 1.1.1
#   Print the equation y = ax + b
print("-" * 200)
print("\t"*1 + "1.1.2 Ridge regression and classification")
print("\t"*2 + "1.1.2.1 Regression")
print("Solve the problem min(w)||Xw-y||²2 + alpha||w||²2")
print("It's a linear regression with sum of squares")
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
print("Create a linear model with a ridge (alpha = 0.5):\n", reg)
print("Train the model with 3 samples of 2 features:\n", reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]))
print("Get the coefficient of the linear regression with ridge:\n", reg.coef_)
print("Get the interception of the linear regression with ridge:\n", reg.intercept_)
print("\t"*2 + "1.1.2.2 Classification")
print("The ridge classifier is fast than logistic regression with a high number of classes. (it can compute projection matrix)")
print("\t"*2 + "1.1.2.3 Setting the regularization parameter: generalized Cross-Validation")
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, -6, 13))
print("Create a linear model with ridge and cross validation:\n", reg)
print("Train the linear model with ridge and cross validation:\n", reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]))
print("Get the alpha value:\n", reg.alpha_)
print("It has the same complexity than ordinary least squares (O(n_samples*n²_features))")
# TODO bonus
#   print the equation y = ax + b + alpha
print("-" * 200)
print("\t"*1 + "1.1.3 Lasso")
print("Lasso is a linear model that estimate sparse coefficients using L1 norm\nThe formula is :\nmin(w)(1/2*n_samples)||Xw-y||^2_2 + alpha * ||w||1\n" + " " * 20 + "with alpha a constant and ||w||1 the L1-norm")
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
print("Creating linear model with lasso using alpha = 0.1:\n", reg)
print("Training linear model with lasso using dataset X_train and prediction y_train:\n", reg.fit([[0, 0], [1, 1]], [0, 1]))
print("Get the score using prediction y_test:\n", reg.predict([[1, 1]]))
print("\t"*2 + "1.1.3.1 Setting regularization parameter")
print("LassoCV formula is (1/ (2 * n_samples)) * ||y - Xw||^2_2 * ||w||_1")
print("For high-dimensional dataset LassoCV is preferable")
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
X, y = make_regression(noise=4, random_state=0)
print("Get dataset X with prediction y:\n", X[:1], y[:5])
reg = LassoCV(cv=5, random_state=0).fit(X, y)
print("Train the model LassoCV with 5-fold CV:\n", reg)
print("Get the score:\n", reg.score(X, y))
print("Get the prediction using the dataset:\n", reg.predict(X[:1, ]))
print("For dataset with more sample than features LassoLarsIC is preferable")
print("-" * 200)
print("\t"*1 + "1.1.4 Multi-task Lasso")
print("It use with L1 and L2 norm")
import numpy as np
from sklearn.linear_model import MultiTaskLasso
rng = np.random.RandomState(42)
n_samples, n_features, n_tasks = 100, 30, 40
print("Prepare the dataset using nb samples, features and tasks:\n:", n_samples, n_features, n_tasks)
n_relevant_features = 5
print("Prepare the nb relevant features:\n", n_relevant_features)
coef = np.zeros((n_tasks, n_features))
print("Initialise the coef:\n", coef)
times = np.linspace(0, 2 * np.pi, n_tasks)
print("Prepare the times:\n", times)
for k in range(n_relevant_features):
    coef[:, k] = np.sin((1. + rng.randn(1, 1)) * times + 3 * rng.randn(1, 1))
print("Fufill the coef using relevant features:\n", coef)
X = rng.randn(n_samples, n_features)
Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
print("Fufill the dataset X and the prediction Y using previous parameters:\n", X[:1], Y[:1])
print("The coef obtains after training of MultiTaskLasso:\n", MultiTaskLasso(alpha=1.).fit(X, Y).coef_)
print("It estimate sparse coefficients for multiple regression problems")
print("-" * 200)
print("\t"*1 + "1.1.5 Elastic-Net")

print("Useful when there are multiple features which are correlated with one another")
print("-" * 200)
print("\t"*1 + "1.1.6 Multi task Elastic-Net")
print("It use mixed L1 and L2-norm and L2-norm for regularization")
print("It estimate sparse coefficients for multiple regression problems too")
print("-" * 200)





