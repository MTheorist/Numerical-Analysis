import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from statistics import mean

# Configure matplotlib to use LaTeX for text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def FDLaplacian2D(hx, hy, Nx, Ny):
    Dx = (1/hx)*sp.diags([[1]*(Nx+1),[-1]*(Nx+1)],[0,-1],shape=(Nx+2,Nx+1))
    Dx = Dx.tolil()
    Dx[0,:] = 0
    Dx[Nx+1,:] = 0
    
    Dy = (1/hy)*sp.diags([[1]*(Ny+1),[-1]*(Ny+1)],[0,-1],shape=(Ny+2,Ny+1))
    Dy = Dx.tolil()
    Dy[0,:] = 0
    Dy[Nx+1,:] = 0

    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    Ix = sp.eye(Nx+1)
    Iy = sp.eye(Ny+1)

    A = sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix)
    return A

def sourcefw(Du, Dv, A, w, k, a, b, Nx, Ny):
    u = w[0]
    v = w[1]
    u_squared = u ** 2
    fu = -Du * A.dot(u) + k * (a - u + u_squared * v)
    fv = -Dv * A.dot(v) + k * (b - u_squared * v)
    fw = np.vstack((fu, fv))
    return fw

def eigenvalue(A, k, T):
    Eigenvalue = la.eigs(A)[0]
    Eigenmax = max(list(Eigenvalue))
    Nt_min = T/(2/(Eigenmax + k))
    return Nt_min

#DOMAIN BOUNDS
LeftX = 0.0
RightX = 4.0
LeftY = 0.0
RightY = 4.0

#COEFFICIENTS OF PDE
Du = 0.05
Dv = 1
k = 5
a = 0.1305
b = 0.7695

#COMPUTATION PARAMETERS
Nx = 100                # number of intervals in x-direction
Ny = 100                # number of intervals in y-direction
hx = (RightX-LeftX)/Nx  # grid step in x-direction
hy = (RightY-LeftY)/Ny  # grid step in y-direction
T = 20                  # maximum time duration
tol = 1e-3              # convergence criteria
Nt_FE = 51000           # number of time steps for Forward Euler
Nt_BE = 50              # number of time steps for Backward Euler
ht_FE = T / Nt_FE       # time steps for Forward Euler
ht_BE = T / Nt_BE       # time steps for Backward Euler

#PERTURBATION TERM
r = 0.01 * (a + b) * np.random.rand((Nx+1)*(Ny+1))

#INITIAL CONDITIONS
u0 = a + b + r
v0 = b / (a + b) ** 2 * np.ones((Nx+1)*(Ny+1))
w0 = np.vstack((u0, v0))

#NEGATIVE FD LAPLACIAN 
A = FDLaplacian2D(hx, hy, Nx, Ny)

#print("Nt required: ", eigenvalue(A, k, T))

#FORWARD EULER TIME INTEGRATION 
T1FE = time.time()
wk = w0
t = 0
while t <= T:
    fw = sourcefw(Du, Dv, A, wk, k, a, b, Nx, Ny)
    wk = wk + (ht_FE * fw)
    t = t + ht_FE
T2FE = time.time()
print("Foward Euler Method", "--- %s seconds ---" % (T2FE - T1FE), "Nt=", Nt_FE)

#BACKWARD EULER NEWTON-RHAPSON TIME INTEGRATION
T1BE = time.time()
wk2 = w0
t, j = 0, 0
fw = sourcefw(Du, Dv, A, wk2, k, a, b, Nx, Ny)

residual = []
while t <= T:
    i = 0
    wk_BE = wk2
    res = np.linalg.norm(wk2 + (ht_BE*fw) - wk_BE)
    while res > tol:
        fw = sourcefw(Du, Dv, A, wk_BE, k, a, b, Nx, Ny)
        u = wk_BE[0]
        v = wk_BE[1]
        u_squared = np.multiply(u,u)
        uv = np.multiply(u,v)
        
        J00 = (-Du*A) + (k*(2 * sp.diags(np.squeeze(uv)) - sp.diags(np.squeeze(np.ones(A.shape[0])))))
        J01 = sp.coo_matrix(k * sp.diags(np.squeeze(u_squared)))
        J10 = sp.coo_matrix(-2 * k * sp.diags(np.squeeze(uv)))
        J11 = sp.coo_matrix((-Dv*A) - (k * sp.diags(np.squeeze(u_squared))))
        J = sp.bmat([[J00, J01], [J10, J11]])
        
        LHS = sp.diags(np.ones(2 * (Nx+1) * (Ny+1))) - ht_BE * J
        RHS = wk2 + (ht_BE*fw) - wk_BE
        p = la.spsolve(LHS, np.reshape(RHS, -1))
        
        wk_BE = wk_BE + p.reshape(2,(Nx+1)*(Ny+1))
        res = np.linalg.norm(wk2 + (ht_BE*fw) - wk_BE)
        i += 1
        if i>10:
            break
        print("Outer iteration: ", j, ", Inner iteration: ", i, ", Residual norm: ", res)
    residual.append(res)
    wk2 = wk_BE
    t = t + ht_BE
    j += 1
T2BE = time.time()

print("BENR, Nt=%s, CPU Time=%s s" % (Nt_BE, str(round(T2BE - T1BE, 2))))
print("residual mean: ", mean(residual))

#PLOTTING ROUTINES
plt.figure(1)   #initial condition- u(x,y,0)
plt.imshow(u0.reshape(Nx+1, Ny+1), extent=([0, 4, 0, 4]))
plt.colorbar()
plt.title("$u(0)$")

plt.figure(2)   #initial condition- v(x,y,0)
plt.imshow(v0.reshape(Nx+1, Ny+1), extent=([0, 4, 0, 4]))
plt.colorbar()
plt.title("$v(0)$")

plt.figure(3)   #Forward Euler for u(x,y,T)
plt.imshow(wk[0].reshape(Nx+1, Ny+1), extent=([0, 4, 0, 4]))
plt.colorbar()
plt.title("$u$ with FE, $Nt=%s$, CPU Time=$%s$ s" % (Nt_FE, str(round(T2FE - T1FE, 2))))

plt.figure(4)   #Forward Euler for v(x,y,T)
plt.imshow(wk[1].reshape(Nx+1, Ny+1), extent=([0, 4, 0, 4]))
plt.colorbar()
plt.title("$v$ with FE, $Nt=%s$, CPU Time=$%s$ s" % (Nt_FE, str(round(T2FE - T1FE, 2))))

plt.figure(5)   #Backward Euler Netwon-Rhapson for u(x,y,T)
plt.imshow(wk2[0].reshape(Nx+1, Ny+1), extent=([0, 4, 0, 4]))
plt.colorbar()
plt.title("$u$ with BENR, $Nt=%s$, CPU Time=$%s$ s" % (Nt_BE, str(round(T2BE - T1BE, 2))))

plt.figure(6)   #Backward Euler Netwon-Rhapson for v(x,y,T)
plt.imshow(wk2[1].reshape(Nx+1, Ny+1), extent=([0, 4, 0, 4]))
plt.colorbar()
plt.title("$v$ with BENR, $Nt=%s$, CPU Time=$%s$ s" % (Nt_BE, str(round(T2BE - T1BE, 2))))