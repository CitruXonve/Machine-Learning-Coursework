
# coding: utf-8

# In[13]:


import numpy as np

def read_data(filename):
    input_f=open(filename, 'r')
    input_data=[]
    for line in input_f.readlines():
        input_data.append([float(val) for val in line.split(',')])
    input_data=np.array(input_data)
    train_x=input_data[:,:-2]
    train_y=input_data[:,-1:]
    train_x=np.concatenate((train_x, np.ones((train_x.shape[0],1))),axis=1)
    n, dimension = train_x.shape
    return n, dimension, train_x, train_y
    
l, dimension, X0, Y0 = read_data('classification.txt')
print X0,Y0

Coefficient = np.zeros((dimension,1))
# Y0=Y0.T

# l = len(X)
# X0 = np.array([np.ones(l), X, Y]).T     # Here I put the first column as all "1"s because the a0 is the intercept, there is no corresponding x
# Coefficient = np.array([0, 0, 0])       # Here are the coefficients. There are 3 entries: the 1st is intercept, the 2nd is X's coefficient and 3rd is Y's coefficient
# Y0 = np.array(Z)                        # Actual value of Z
# X0


# In[8]:
def activation(val):
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            val[i,j]= 1.0 if val[i,j] > 0.0 else -1.0
    return val

## Here is the cost function:
#  J=sigma(h0(xi)-yi)^2/2m
## The gradient is the partial derivative of J: gradient= sigma(h0(xi)-yi)*xi
## Then we update the coefficient every iteration.
def gradient_descent(X, Y, C, learning_rate, iterations):
    l = len(Y)    
    for iteration in range(iterations):
        H = activation(X.dot(C))  # H is the hypothesis value (X bar)
        print X,C,H 
        delta_x = H - Y # delta_x is the difference between hypothesis value and actural value of Z
        gradient = X.T.dot(delta_x) / l  # Here is the gradient   
        C = C - learning_rate * gradient   # We update the coefficient by subtracting learning rate multipled by the partial derivative of cost func
    return iteration, C

# 7000 Iterations with learning rate of 0.001
iterarion, Coefficients = gradient_descent(X0, Y0, Coefficient, 0.001, 5)

# Intercept a0, Coefficient of X: a1, Coefficient of Y:a2
print 'Epoch #', iterarion , 'Weight matrix=',Coefficients.T
print activation(np.dot(X0, Coefficient))-Y0

