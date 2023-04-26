'''
import numpy as np

x = np.linspace(-np.pi, np.pi,200)
y = np.linspace(-np.pi, np.pi,200)
XY = np.meshgrid(x,y)

def matrix(x,y):

    F = np.exp(y*1j) + 2*np.exp(-1j*y/2)* np.cos(np.sqrt(3)/2*x)
    Fc = np.exp(-y*1j) + 2*np.exp(1j*y/2)* np.cos(np.sqrt(3)/2*x)
    return np.matrix('0 F;Fc 0')

for i in range(200):
    for j in range(200):
        matrix(i,j)
'''

import numpy as np

def matrix(x,y):
    F = np.exp(y*1j) + 2*np.exp(-1j*y/2)* np.cos(np.sqrt(3)/2*x)
    Fc = np.exp(-y*1j) + 2*np.exp(1j*y/2)* np.cos(np.sqrt(3)/2*x)
    return np.matrix([[0, F], [Fc, 0]])

# Generate a grid of x and y values
x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100))

# Evaluate M for each (x,y) point
M_values = np.zeros((100, 100, 2, 2), dtype=complex)
eigenvalues = np.zeros((100, 100, 2), dtype=complex)
for i in range(100):
    for j in range(100):
        M_values[i, j] = matrix(x[i, j], y[i, j])
        eigenvalues[i, j] = np.linalg.eig(M_values[i, j])
# Diagonalize M for each (x,y) point



# Plot the real parts of the eigenvalues as a function of (x,y)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.contourf(x, y, np.real(eigenvalues[:,:,1]), cmap='RdBu')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()