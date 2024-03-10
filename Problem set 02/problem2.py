import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def newton_multi_var(grad_F, H, x_init, fixed_step):
    x = x_init
    iter = 0
    while True:
        iter = iter + 1
        H_inv = np.linalg.inv(np.float64(H(x)))
        x_move = np.dot(H_inv, np.float64(grad_F(x)))
        x = x - fixed_step*x_move
        if np.linalg.norm(x_move) < 1e-8:
            break

    return x, iter

w1, w2 = symbols("w1 w2")

f = w1**2 + w2**2 + (0.5*w1 + w2)**2 + (0.5*w1 + w2)**4
x_init = np.array([3, 3])
fixed_step = 1
grad_w1 = f.diff(w1)
grad_w2 = f.diff(w2)
calculate_grad_f = lambda val: np.array(
    [
        grad_w1.subs([(w1, val[0]), (w2, val[1])]),
        grad_w2.subs([(w1, val[0]), (w2, val[1])]),
    ]
)
H = lambda val: np.array(
    [
        [
            f.diff(w1, w1).subs([(w1, val[0]), (w2, val[1])]),
            f.diff(w1, w2).subs([(w1, val[0]), (w2, val[1])]),
        ],
        [
            f.diff(w1, w2).subs([(w1, val[0]), (w2, val[1])]),
            f.diff(w2, w2).subs([(w1, val[0]), (w2, val[1])]),
        ],
    ]
)

local_min, iterations = newton_multi_var(calculate_grad_f, H, x_init, fixed_step)
round_local_min_pos = np.array([round(local_min[0], 3), round(local_min[1], 3)])

print("Number of iteration: ", iterations)
print("Local min: ({0},{1})".format(round_local_min_pos[0], round_local_min_pos[1]))
f = lambda val: val[0]**2 + val[1]**2 + (0.5*val[0] + val[1])**2 + (0.5*val[0] + val[1])**4

x = np.linspace(-10, 10, 10)
y = np.linspace(-10, 10, 10)
X, Y = np.meshgrid(x, y)
Z = f(np.array([X, Y]))
ax = plt.axes(projection="3d")
ax.set_title("f = w1**2 + w2**2 + (0.5*w1 + w2)**2 + (0.5*w1 + w2)**4")
ax.contour3D(X, Y, Z, 30)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.scatter([local_min[0]], [local_min[1]], [f(local_min)], color="red")

label = "Local min: ({0},{1})".format(round_local_min_pos[0], round_local_min_pos[1])

ax.text(round_local_min_pos[0], round_local_min_pos[1], f(local_min), label, None)

plt.show()