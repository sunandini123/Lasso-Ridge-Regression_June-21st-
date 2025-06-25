
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#import linear regression ML algorithm 
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge  # ✅ Import Ridge
from sklearn.linear_model import Lasso



data =pd.read_csv(r'/Users/shashi/Desktop/car-mpg.csv')
data.head()


#drop car name
#replace origin into 1,2,3 ... dont forget to get dummies
#replace ? with nan
#replace all nan values with median
data=data.drop(['car_name'],axis=1)
data['origin']=data['origin'].replace({1:'america',2:'europe',3:'asia'})
data=pd.get_dummies(data,columns=['origin'],dtype=int)
data=data.replace('?',np.nan)


# Fill missing values with median only for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].apply(lambda x: x.fillna(x.median()))


# This splits your DataFrame data into:

#X: The features (all columns except mpg).

# y: The target variable (only the mpg column).
X=data.drop(['mpg'],axis=1)
y=data[['mpg']]

X_s=preprocessing.scale(X)
X_s=pd.DataFrame(X_s,columns=X.columns)


y_s=preprocessing.scale(y)
y_s=pd.DataFrame(y_s,columns=y.columns)

# To remove a single Nan value in the dataset
X_s['hp'] = X_s['hp'].fillna(X_s['hp'].median())
X_s
y_s
X_train,X_test,y_train,y_test=train_test_split(X_s,y_s,test_size=0.20,random_state=0)
X_train.shape


#2.a Simple Linear MOdel
#Fit Simple linear model and find coefficients
regression_model=LinearRegression()
regression_model.fit(X_train,y_train)
for idx, col_name in enumerate(X_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))


intercept=regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))

#2.b Regularized Ridge Regression
#alpha factor here is lambda(penalty term) which helps to reduce the magnitude of coeff
ridge_model=Ridge(alpha=0.3)
ridge_model.fit(X_train,y_train)
# Creates a Ridge Regression model with regularization strength alpha=0.3.
# Fits (trains) that model on your training data X_train (features) and y_train (target).



print('Ridge model coef: {}'.format(ridge_model.coef_))
#as the data has 10 columns hence 10 coefficients appear here




#2.c Regularized Lasso Regression
lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)


print('Lasso model coef: {}'.format(lasso_model.coef_))


#3 Score Comparison
#Model score - r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS) = Regression error/TSS

#Simple Linear Model
print(regression_model.score(X_train, y_train))
print(regression_model.score(X_test, y_test))


print('*************************')
#Ridge
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))

print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))


#Model Parameter Tuning
data_train_test=pd.concat([X_train,y_train],axis=1)
data_train_test.head()

import statsmodels.formula.api as smf

# Build and fit the OLS model in one go:
ols1 = smf.ols(
    formula='mpg ~ cyl + disp + hp + wt + acc + yr + car_type + '
            'origin_america + origin_europe + origin_asia',
    data=data
).fit()

# Now you can inspect coefficients and summary:
print(ols1.params)
print(ols1.summary())

#Lets check Sum of Squared Errors (SSE) by predicting value of y for test cases and subtracting from the actual y for the test cases
mse = np.mean((regression_model.predict(X_test)-y_test)**2)


# root of mean_sq_error is standard deviation i.e. avg variance between predicted and actual
import math
rmse = math.sqrt(mse)
print('Root Mean Squared Error: {}'.format(rmse))

# Is OLS a good model ? Lets check the residuals for some of these predictor.
fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['hp'], y= y_test['mpg'], color='green', lowess=True )

fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['acc'], y= y_test['mpg'], color='green', lowess=True )

# predict mileage (mpg) for a set of attributes not in the training or test set
y_pred = regression_model.predict(X_test)



# Since this is regression, plot the predicted y value vs actual y values for the test data
# A good model's prediction will be close to actual leading to high R and R2 values
#plt.rcParams['figure.dpi'] = 500
plt.scatter(y_test['mpg'], y_pred)


















