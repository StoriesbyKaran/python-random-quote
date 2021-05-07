#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics Internship
# 
# ## Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 
# ### Author: Karan Babu Daroga
# 
# **Not for distribution.**

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("Libraries imported successfully")


# ## Reading the data from github repository

# In[7]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
print("Data imported successfully")


# ## Data Exploration

# In[9]:


data.head()


# In[10]:


data.tail()


# In[13]:


#Checing for any missing values 

data.isnull().sum()


# In[14]:


data.describe()


# In[15]:


data.info()


# In[20]:


#Checking the Correlation between the given Hours and Scores

data.corr()


# This shows higher correlation between Hours and Scores

# ## Data Visualization 

# In[21]:


#Visualizing with scatter plot 

data.plot(x='Hours', y='Scores', style='*',color='red')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the above graph we can say that their is a positive linear relationship between the number of hours studied and scores.
# 
# 
# 
# 

# ## Data Modelling 
# 
# We now divide the data into attributes and labels. Attributes are the independent variables and labels are dependent variables. Dependent variables are the ones whose values are predicted. From our dataset we want to predict the percentage score for the hours studied. Hence attributes will be the "Hours" column and labels will be the "Score" column.

# In[22]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[23]:


X


# In[24]:


y


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method.  The below code will split 80 % of the data to training data set and 20 % of the data to the test data set

# In[25]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=2)


# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[27]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training completed.")
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)


# Plotting the fit line over the data in single linear regression 

# In[28]:


line = regressor.coef_*X+regressor.intercept_
plt.title("Linear regression vs trained model")
plt.scatter(X, y,color='red')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.plot(X, line);
plt.show()


# Blue line is the best fit line for this data

# ## Predictions
# 
# After training our algorithms. We will now make the predictions. Y pred is a numpy array that contains all the predicted values for the input values in the X_test series

# In[29]:


print(X_test) 
y_pred = regressor.predict(X_test)


# In[30]:


y_pred


# In[31]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[32]:


df.plot(kind='bar')


# In[33]:


hours = 9.25
test=np.array([hours])
test=test.reshape(-1,1)
pred = regressor.predict([[9.5]])
print("Hours studied = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# ## Model Evaluation 
# 
# There are 3 main metrics for model evaluation in regression:
# 1. R Square/Adjusted R Square
# 2. Mean Square Error(MSE)/Root Mean Square Error(RMSE)
# 3. Mean Absolute Error(MAE)
# 
# Using metrics to find the mean absolute error and r2 to find the accuracy of the model
# 
# R Square/Adjusted R Square are better used to explain the accuracy because we can explain how well the regression predictions approximate the real data points. MSE, RMSE or MAE are better to be used to compare performance between different regression models

# In[35]:


from sklearn import metrics
from sklearn.metrics import r2_score
print("R2_Score: %.2f" % r2_score(y_test,y_pred))


# Higher the r2 value, higher is the acccuracy of the model. r2 ranges from 0-1.

# In[36]:


print('Mean Absolute Error Is : ' , metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error Is : ' , metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error Is : ' , np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))


# In[ ]:




