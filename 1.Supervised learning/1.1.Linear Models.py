print("1.1 Linear Models")
print("Linear combination of the features with y as the predicted value\nThe formula is :\n" + " " * 15 + " y(w, x) = w0 + w1x1 + ... + wpxp\n" + " " * 20 + "with w the interception(difference) and x the dataset")
print("-"*500)
print("\t"*1 + "1.1.1 Ordinary Least Squares")
print("Solve the problem min(w)=||Xw - y||²2")
from sklearn import linear_model
reg = linear_model.LinearRegression()
print("Create the linear regression object:\n", reg)
print("Train the linear regression object:\n", reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2]))
print("Get the coefficient of the linear regression:\n", reg.coef_)
# TODO bonus 1.1.1
#   Print the equation y = ax + b
print("-" * 200)
