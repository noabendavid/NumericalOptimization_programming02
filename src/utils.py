import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_qp(result, path_history):

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Quadratic function', fontsize=16)

    # First subplot: 3D path
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot feasible region
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    Z[Z < 0] = np.nan
    ax1.plot_surface(X, Y, Z, alpha=0.5, color='yellow')

    # Plot path
    path = np.array(path_history['path'])
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', label='Path')

    # Plot final candidate
    ax1.scatter(result[0], result[1], result[2], color='red', s=70, label='Final candidate')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Path taken by the algorithm in 3D space')
    ax1.view_init(45, 45)
    ax1.legend()

    # Second subplot: Objective value vs outer iteration number
    ax2 = fig.add_subplot(122)
    values = path_history['values']
    iterations = np.arange(len(values))
    ax2.plot(iterations, values, marker='.')
    ax2.set_xlabel('Outer iteration number')
    ax2.set_ylabel('Objective value')
    ax2.set_title('Objective value vs Outer iteration number')

    plt.tight_layout()
    plt.show()

def plot_lp(result, path_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Linear function', fontsize=16) 
    
    # Define the constraints
    x = np.linspace(-1, 3, 1000)
    y1 = -x + 1
    y2 = np.ones_like(x)
    y3 = np.ones_like(x) * 0

    # Plot the constraints on the first subplot
    ax1.plot(x, y1, label='y ≥ -x + 1')
    ax1.plot(x, y2, label='y ≤ 1')
    ax1.axvline(x=2, color='g', label='x ≤ 2')
    ax1.axhline(y=0, color='m', label='y ≥ 0')

    # Fill the feasible region
    ax1.fill([0, 2, 2, 1], [1, 1, 0, 0], color='lightgray', label='Feasible region')

    # Plot path
    path = np.array(path_history['path'])
    ax1.plot(path[:, 0], path[:, 1], marker='o', label='Path', color='yellow')

    # Plot final candidate
    ax1.scatter(result[0], result[1], color='red', s=100, label='Final candidate')

    ax1.set_xlim([-1, 3])
    ax1.set_ylim([-1, 2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Path taken by the algorithm in 2D space')

    ax1.legend()

    # Plot objective value vs outer iteration number on the second subplot
    values = path_history['values']
    iterations = np.arange(len(values))
    ax2.plot(iterations, values, marker='.')
    ax2.set_xlabel('Outer iteration number')
    ax2.set_ylabel('Objective value')
    ax2.set_title('Objective value vs outer iteration number')

    plt.tight_layout()
    plt.show()