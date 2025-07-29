import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse.linalg as linalg
import torch
import torch_sparse
import torch.linalg
import random
plt.rcParams["font.family"] = "Arial"
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

def H_kohei_Lx_Ly(Lx, Ly, t, m_vals, gamma):
    H = np.zeros((2*Lx*Ly, 2*Lx*Ly), dtype=np.complex64)
    
    for y in range(Ly):
        for x in range(Lx-1):
            H[2*Lx*y + 2*x:2*Lx*y + 2*x+2, 2*Lx*y + 2*(x+1):2*Lx*y + 2*(x+1)+2] = t[y,x]*t[y,x+1]/2*(sx+1j*sy).conjugate().transpose()
            H[2*Lx*y + 2*(x+1):2*Lx*y + 2*(x+1)+2, 2*Lx*y + 2*x:2*Lx*y + 2*x+2] = (t[y,x]*t[y,x+1]/2*(sx+1j*sy))
        
        # H[2*Lx*y + 2*0:2*Lx*y + 2*0+2, 2*Lx*y + 2*Lx-2:2*Lx*y + 2*Lx] = 1/2*(sx+1j*sy) 
        # H[2*Lx*y + 2*Lx-2:2*Lx*y + 2*Lx, 2*Lx*y + 2*0:2*Lx*y + 2*0+2] = 1/2*(sx+1j*sy).conjugate().transpose()
            
    for x in range(Lx):
        for y in range(Ly-1):
            H[2*Lx*y + 2*x:2*Lx*y + 2*x+2, 2*Lx*(y+1) + 2*x:2*Lx*(y+1) + 2*x+2] = t[y,x]*t[y+1,x]/2*(sx+1j*sz)
            H[2*Lx*(y+1) + 2*x:2*Lx*(y+1) + 2*x+2, 2*Lx*y + 2*x:2*Lx*y + 2*x+2] = (t[y,x]*t[y+1,x]/2*(sx+1j*sz)).conjugate().transpose()
        
        # H[2*x:2*x+2, 2*Lx*(Ly-1)+2*x:2*Lx*(Ly-1)+2*x+2] = (1/2*(sx+1j*sz)).conjugate().transpose()
        # H[2*Lx*(Ly-1)+2*x:2*Lx*(Ly-1)+2*x+2, 2*x:2*x+2] = 1/2*(sx+1j*sz)
        
    for x in range(Lx):
        for y in range(Ly):
            H[2*Lx*y + 2*x:2*Lx*y + 2*x+2, 2*Lx*y + 2*x:2*Lx*y + 2*x+2] = m_vals[y,x]*sx + 1j * gamma[y,x] * sy
    
    return sp.csr_matrix(H)

print('single layer Hamiltonian done')

def H_four_layer_Lx_Ly_all(Lx, Ly, t, m_vals, gamma, T, Ts, To):
    
    H1 = H_kohei_Lx_Ly(Lx, Ly, t, m_vals, gamma)
    H2 = H_kohei_Lx_Ly(Lx, Ly, t, m_vals, -gamma)
    H3 = H_kohei_Lx_Ly(Lx, Ly, t, -m_vals, -gamma)
    H4 = H_kohei_Lx_Ly(Lx, Ly, t, -m_vals, gamma)
    
    size = 4*2*Lx*Ly
    H_4_layer = sp.lil_matrix((size, size), dtype=np.complex64)
    
    row_slice_1 = np.s_[0:2*Lx*Ly]
    H_4_layer[row_slice_1, row_slice_1] = H1

    row_slice_2 = np.s_[2*Lx*Ly:4*Lx*Ly]
    H_4_layer[row_slice_2, row_slice_2] = H2

    row_slice_3 = np.s_[4*Lx*Ly:6*Lx*Ly]
    H_4_layer[row_slice_3, row_slice_3] = H3

    row_slice_4 = np.s_[6*Lx*Ly:8*Lx*Ly]
    H_4_layer[row_slice_4, row_slice_4] = H4


    m_prime = np.reshape(m_vals, (1, Lx*Ly))
    m_prime = m_prime * np.eye(Lx*Ly)
    m_prime = np.kron(m_prime, np.eye(2))

    H_4_layer[row_slice_1, row_slice_2] = T * m_prime
    H_4_layer[row_slice_2, row_slice_1] = T * m_prime
    
    H_4_layer[row_slice_3, row_slice_4] = T * m_prime
    H_4_layer[row_slice_4, row_slice_3] = T * m_prime
    
    H_4_layer[row_slice_2, row_slice_3] = To * m_prime
    H_4_layer[row_slice_3, row_slice_2] = To * m_prime
    
    H_4_layer[row_slice_1, row_slice_4] = To * m_prime
    H_4_layer[row_slice_4, row_slice_1] = To * m_prime
    
    H_4_layer[row_slice_1, row_slice_3] = Ts * m_prime
    H_4_layer[row_slice_3, row_slice_1] = Ts * m_prime
    
    H_4_layer[row_slice_2, row_slice_4] = Ts * m_prime
    H_4_layer[row_slice_4, row_slice_2] = Ts * m_prime
    
    return H_4_layer.tocsr()

print('4 layer Hamiltonian done')


f0 = np.zeros((3,3))
f0[1,2] = 1
f0[2,1] = 1
f0[0,1] = 1
f0[1,0] = 1
f0[1,1] = 1
# plt.imshow(f0, cmap='gray')

# Fractal generation and visualization (unchanged)
def fractal(n):
    if n == 0:
        return f0
    else:
        return np.kron(fractal(n-1), f0)

print('fractal done')


def wavepacket_four_layer(psi0, t0, tmax, dt, H):
    def f(x, time):
        return -1j * np.dot(H, x.T)

    time = np.arange(t0, tmax, dt)
    nsteps = len(time)
    psi = np.zeros((nsteps, len(H)), dtype=np.complex_)
    psi[0, :] = psi0
    
    for i in range(1, nsteps):
        k1 = f(psi[i-1, :], time[i-1])
        k2 = f(psi[i-1, :] + 0.5*dt*k1, time[i-1] + 0.5*dt)
        k3 = f(psi[i-1, :] + 0.5*dt*k2, time[i-1] + 0.5*dt)
        k4 = f(psi[i-1, :] + dt*k3, time[i-1] + dt)
        psi[i, :] = psi[i-1, :] + \
            (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return psi, time, nsteps



print('wavepacket done')