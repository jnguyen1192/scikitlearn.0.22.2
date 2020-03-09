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
print("Lasso is a linear model that estimate sparse coefficients using L1 norm\nThe formula is :\nmin(w)(1/2*n_samples)||Xw-y||²2 + alpha||w||1\n" + " " * 20 + "with alpha a constant and ||w||1 the L1-norm")
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
print("Creating linear model with lasso using alpha = 0.1:\n", reg)
print("Training linear model with lasso using dataset X_train and prediction y_train:\n", reg.fit([[0, 0], [1, 1]], [0, 1]))
print("Get the score using prediction y_test:\n", reg.predict([[1, 1]]))
print("\t"*2 + "1.1.3.1 Setting regularization parameter")
print("For high-dimensional dataset LassoCV is preferable")
print("For dataset with more sample than features LassoLarsIC is preferable")
print("-" * 200)
print("\t"*1 + "1.1.3 Multi-task Lasso")
print("It use with L1 and L2 norm")
print("It estimate sparse coefficients for multiple regression problems")
print("-" * 200)
print("\t"*1 + "1.1.4 Elastic-Net")
print("Useful when there are multiple features which are correlated with one another")





