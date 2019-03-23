
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[14]:


data = pd.read_csv('linear-regression.txt',names=["X","Y","Z"])      
data.head()


# In[15]:


X = data['X'].values
Y = data['Y'].values
Z = data['Z'].values
# X and Y are the independent variables and Z is the dependent variable 
# Z=a0+a1X+a2Y


# In[16]:


# In this implementation, I use gradient descent algorithm:
# The cost function J(a0,a1,a2) is computed and I update the coefficient a0,a1,a2 based 
# on the partial derivative of cost function J every iteration. The updating equation is:
# C=c - learning rate* d/dax(J).
# I predefine the learning rate as 0.001 and set iteration 7000 times.

l = len(X)
X0 = np.array([np.ones(l), X, Y]).T     # Here I put the first column as all "1"s because the a0 is the intercept, there is no corresponding x
Coefficient = np.array([0, 0, 0])       # Here are the coefficients. There are 3 entries: the 1st is intercept, the 2nd is X's coefficient and 3rd is Y's coefficient
Y0 = np.array(Z)                        # Actual value of Z
X0


# In[8]:


## Here is the cost function:
#  J=sigma(h0(xi)-yi)^2/2m
## The gradient is the partial derivative of J: gradient= sigma(h0(xi)-yi)*xi
## Then we update the coefficient every iteration.
def gradient_descent(X, Y, C, learning_rate, iterations):
    l = len(Y)    
    for iteration in range(iterations):
        H = X.dot(C)  # H is the hypothesis value (X bar) 
        delta_x = H - Y # delta_x is the difference between hypothesis value and actural value of Z
        gradient = X.T.dot(delta_x) / l  # Here is the gradient   
        C = C - learning_rate * gradient   # We update the coefficient by subtracting learning rate multipled by the partial derivative of cost func
        # if np.sum(delta_x**2)<1.0:
        #     break
    
    return C


# In[24]:




# 7000 Iterations with learning rate of 0.001
Coefficients = gradient_descent(X0, Y0, Coefficient, 0.001, 7000)

# Intercept a0, Coefficient of X: a1, Coefficient of Y:a2
print(Coefficients)

