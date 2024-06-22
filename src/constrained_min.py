import numpy as np
import math

def log_barrier(ineq_constraints, x0):
    log_f = 0
    log_g = np.zeros_like(x0)
    log_h = np.zeros((x0.size, x0.size))

    # for constraint inineq_constraints :
        # f, g, h = constraint(x0)
    constraint = ineq_constraints(x0)

    for item in range(len(constraint)):
        f, g, h = constraint[item]
        inv_f = -1.0 / f
        log_f += math.log(-f)
        log_g += inv_f * g

        grad_tile = np.outer(g, g) / f ** 2
        log_h += (h * f - grad_tile) / f ** 2
    
    return -log_f, log_g, -log_h

def applay_log_barrier(f, ineq_constraints, x0, t):
    
    func_val, grad, hess = f(x0)
    log_f, log_g, log_h = log_barrier(ineq_constraints, x0)

    #combine func with  barrier terms 
    func_res = t*func_val + log_f
    grad_res = t*grad + log_g
    hess_res = t*hess + log_h

    return func_res, grad_res, hess_res

def wolfe_condition_backtracking(f, x, val, grad, direction, c1=0.01, beta=0.5, max_iter=10):   
    alpha = 1
    cval, _, _ = f(x + alpha * direction)
    dot_th = c1 * np.dot(grad, direction)
    
    idx = 0 
    while idx < max_iter and  cval > val + alpha * dot_th: 
        alpha *= beta
        cval, _, _ = f(x + alpha * direction)
        idx += 1
    return alpha 

def newton_step(eq_constraints_mat, eq_constraints_rhs, hess, grad):
 
    # Solve the KKT system
    A = eq_constraints_mat
    b = eq_constraints_rhs

    if A.size == 0:
        return np.linalg.solve(hess,-grad)
    else:
        KKT_matrix = np.block([[hess, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
        KKT_rhs = np.hstack([-grad, np.zeros(A.shape[0])])
        delta = np.linalg.solve(KKT_matrix, KKT_rhs)
        return delta[:A.shape[1]]


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, t0=1.0, mu=10, tol=1e-8, max_iter=10):
    #Initials   
    n = len(ineq_constraints(x0)) 
    t = t0
    x = x0
    gap = n / t
        
    path_history = {'path': [], 'values': []}
    path_history['path'].append(x.copy())
    path_history['values'].append(f(x)[0])

    while gap > tol:
        for _ in range(max_iter):
            f_x, grad_x, hess_x = applay_log_barrier(f, ineq_constraints, x, t)
            direction = newton_step(eq_constraints_mat, eq_constraints_rhs, hess_x, grad_x)
            steps = wolfe_condition_backtracking(f, x, f_x, grad_x, direction)

            x = x + direction * steps
            th = np.sqrt(np.dot(direction, np.dot(hess_x, direction.T)))

            if 0.5 * (th ** 2) < tol:
                break

        path_history['path'].append(x.copy())
        path_history['values'].append(f(x)[0])
        t *= mu
        gap = n / t


    return x, f(x)[0], path_history
 
 
  