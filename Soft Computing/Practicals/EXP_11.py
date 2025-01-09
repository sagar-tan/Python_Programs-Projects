#Write a program to plot various membership functions. (i) Triangular membership function (ii) Trapezoidal membership function (iii) Bell-shaped membership function
import numpy as np
import matplotlib.pyplot as plt
def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))
def trapezoidal_mf(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a), 1), (d-x)/(d-c)))
def bell_shaped_mf(x, a, b, c):
    return 1 / (1 + np.abs((x-c)/a)**(2*b))
x = np.linspace(0, 10, 1000)
tri_mf = triangular_mf(x, 2, 5, 8)
plt.plot(x, tri_mf, label='Triangular MF')
trap_mf = trapezoidal_mf(x, 2, 4, 6, 8)
plt.plot(x, trap_mf, label='Trapezoidal MF')
bell_mf = bell_shaped_mf(x, 2, 4, 5)
plt.plot(x, bell_mf, label='Bell-shaped MF')
plt.title('Membership Functions')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.show()
