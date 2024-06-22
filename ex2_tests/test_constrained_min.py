import unittest
import numpy as np
from src.constrained_min import interior_pt 
from src.utils import * 
from ex2_tests.examples import *

def main_run(name, x0, func, eq_constraint, ineq_constraint, graph):

    print('*'*20 + f'\n ***** {name} *****\n' + ('*'*20 ))

    eq_constraints_mat, eq_constraints_rhs = eq_constraint()
    
    candidate, objective, path_history = interior_pt(func, ineq_constraint, eq_constraints_mat, eq_constraints_rhs, x0)
    candidate_ineq_containt = [ineq[0] for ineq in ineq_constraint(candidate)]

    print(f"{name} Optimal solution:", candidate)
    print(f"{name} Optimal value:", objective)
    print(f"{name} Inequality Path history:", candidate_ineq_containt)
    if eq_constraints_mat.size > 0:
        print(f'{name} Equality value:', (eq_constraints_mat * candidate).sum())

    graph(candidate,path_history)



class TestMinimization(unittest.TestCase):
    def test_qp(self):
        main_run('QP', np.array([0.1, 0.2, 0.7]),qp_function,qp_eq_constraints,qp_ineq_constraints, plot_qp)

    def test_lp(self):
        main_run('LP', np.array([0.5, 0.75]), lp_function, lp_eq_constraints, lp_ineq_constraints, plot_lp)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
