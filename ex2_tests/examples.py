import numpy as np

# Quadratic Function (QP)
def qp_function(x):

    func = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad =  np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
    hess =  np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    
    return func, grad, hess

def qp_ineq_constraints(x):

    ineq1 = (-x[0], np.array([-1, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])) # x>= 0 
    ineq2 = (-x[1], np.array([0, -1, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])) # y>= 0
    ineq3 = (-x[2], np.array([0, 0, -1]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])) #z>= 0 
    
    return [ineq1, ineq2, ineq3]

def qp_eq_constraints():
    # x+ y + z = 1 
    A = np.array([[1, 1, 1]])
    b = np.array([1])
    return(A, b)


# Linear function (LP) 
def lp_function(x):
    
    func = -x[0] - x[1]
    grad = np.array([-1, -1])
    hess = np.array([[0, 0], [0, 0]])
    
    return func, grad, hess

def lp_ineq_constraints(x):

    ineq1 = (- x[1] - x[0] + 1, np.array([-1, -1]), np.array([[0, 0], [0, 0]]))  # y >= -x + 1
    ineq2 = (x[1] - 1, np.array([0, 1]), np.array([[0, 0], [0, 0]])) # y <= 1
    ineq3 = (x[0] - 2, np.array([1, 0]), np.array([[0, 0], [0, 0]])) # x <= 2
    ineq4 = (-x[1], np.array([0, -1]), np.array([[0, 0], [0, 0]])) # y >= 0

    return [ineq1, ineq2, ineq3, ineq4]

def lp_eq_constraints():
    A = np.array([])
    b = np.array([])
    return(A, b) 