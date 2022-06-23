import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like 
         sigmoid(z)
    """
    # (≈ 1 line of code)
    # s = 
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1 + np.exp(-z))
    ### END CODE HERE ###
    
    return s


def compute_cost(X, y, w, b): 
    m = X.shape[0]

    f_w = sigmoid(np.dot(X, w) + b)
    total_cost = (1/m)*np.sum(-y*np.log(f_w) - (1-y)*np.log(1-f_w))
    
    return float(np.squeeze(total_cost))

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for logistic regression.
    
    Parameters
    ----------
    X : array_like
        Shape (m, n+1) 
    
    y : array_like
        Shape (m,) 
    
    w : array_like
        Parameters of the model
        Shape (n+1,)
    b:  scalar
    
    Returns
    -------
    dw : array_like
        Shape (n+1,)
        The gradient 
    db: scalar
        
    """
    m = X.shape[0]
    f_w = sigmoid(np.dot(X, w) + b)
    err = (f_w - y)
    dw = (1/m)*np.dot(X.T, err)
    db = (1/m)*np.sum(err)
    
    return float(np.squeeze(db)), dw


def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters theta
    
    Parameters
    ----------
    X : array_like
        Shape (m, n+1) 
    
    w : array_like
        Parameters of the model
        Shape (n, 1)
    b : scalar
    
    Returns
    -------

    p: array_like
        Shape (m,)
        The predictions for X using a threshold at 0.5
        i.e. if sigmoid (theta.T*X) >=0.5 predict 1
    """
    
    # number of training examples
    m = X.shape[0]   
    p = np.zeros(m)
   
    for i in range(m):
        f_w = sigmoid(np.dot(w.T, X[i]) + b)
        p[i] = f_w >=0.5
    
    return p

def compute_cost_reg(X, y, w, b, lambda_=1): 
    """
    Computes the cost for logistic regression
    with regularization
    
    Parameters
    ----------
    X : array_like
        Shape (m, n+1) 
    
    y : array_like
        Shape (m,) 
    
    w: array_like
        Parameters of the model
        Shape (n+1,)
    b: scalar
    
    Returns
    -------
    cost : float
        The cost of using theta as the parameter for logistic 
        regression to fit the data points in X and y
        
    """
    # number of training examples
    m = X.shape[0]
    
    # You need to return the following variables correctly
    cost = 0

    f = sigmoid(np.dot(X, w) + b)
    reg = (lambda_/(2*m)) * np.sum(np.square(w))
    cost = (1/m)*np.sum(-y*np.log(f) - (1-y)*np.log(1-f)) + reg
    return cost


def compute_gradient_reg(X, y, w, b, lambda_=1): 
    """
    Computes the  gradient for logistic regression
    with regularization
    
    Parameters
    ----------
    X : array_like
        Shape (m, n+1) 
    
    y : array_like
        Shape (m,) 
    
    w : array_like
        Parameters of the model
        Shape (n+1,)
    b : scalar
    
    Returns
    -------
    db: scalar
    dw: array_like
        Shape (n+1,)

    """
    # number of training examples
    m = X.shape[0]
    
    # You need to return the following variables correctly
    cost = 0
    dw = np.zeros_like(w)

    f = sigmoid(np.dot(X, w) + b)
    err = (f - y)
    dw = (1/m)*np.dot(X.T, err)
    dw += (lambda_/m)  * w
    db = (1/m) * np.sum(err)
 
    #print(db,dw)

    return db,dw
