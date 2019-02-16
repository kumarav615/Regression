# Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_set = pd.read_csv('Position_Salaries.csv')
# upper bound of range is executed but X is converted to matrix else it
# would be treated as a vector
X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2].values

# Fitting linear regression to the data set

regressor1 = LinearRegression()
regressor1.fit(X, y)

# Fitting Polynomial Regression to data set

# this degree value define how many polynomial is added
# x square, x cube, x to power 4 etc
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
regressor2 = LinearRegression()
regressor2.fit(X_poly, y)

# Visualising the Linear Regression Results
plt.scatter(X, y, color='red')
plt.plot(X, regressor1.predict(X), color='blue')
plt.title('Check Truth on Salary(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
# below we can use X_poly but this is not make it generic accepting new
# values for X, so instead make it for any X values
plt.plot(X_grid, regressor2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Check Truth on Salary(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
regressor1.predict(6.5)

# Predicting a new result with Polynomial Regression
regressor2.predict(poly_reg.fit_transform(6.5))
