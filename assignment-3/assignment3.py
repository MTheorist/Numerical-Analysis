import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

def create2DLFVM(hx, hy, Nx, Ny, coeffFun):

    D1 = [] # principal diagonal
    D2 = [] # offset diagonal for 1/hx
    D3 = [] # offset diagonal for 1/hy

    for j in range(1, Ny):
        for i in range(1, Nx):
            D1 += [coeffFun(hx * (i - 0.5), hy * j) / hx ** 2 + coeffFun(hx * i, hy * (j - 0.5)) / hy ** 2 + coeffFun(hx * (i + 0.5), hy * j) / hx ** 2 + coeffFun(hx * i, hy * (j + 0.5)) / hy ** 2]
            if np.size(D2) != ((Nx - 1) * (Ny - 1) - 1):
                if i == Nx - 1:
                    D2 += [0]
                else:
                    D2 += [-coeffFun(hx * (i + 0.5), hy * j) / hx ** 2]

            if j != Ny - 1:
                D3 += [-coeffFun(hx * i, hy * (j + 0.5)) / hy ** 2]

    A = sp.diags([D1, D2, D2, D3, D3], [0, -1, 1, -(Nx-1), Nx-1], format='csc')

    return A

def FDLaplacian2D(hx, hy, Nx, Ny):
    Dx = (1/hx)*sp.diags([[1]*(Nx-1),[-1]*(Nx-1)],[0,-1],shape=(Nx,(Nx-1)))
    Dy = (1/hy)*sp.diags([[1]*(Ny-1),[-1]*(Ny-1)],[0,-1],shape=(Ny,(Ny-1)))

    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    Ix = sp.eye(Nx-1)
    Iy = sp.eye(Ny-1)

    A = sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix)
    return A

def sourcefunc(x,y):
    f = 0
    a = 40
    for i in range(1,10):
        for j in range(1,5):
            f += np.exp(-((a*(x-i)**2)+(a*(y-j)**2)))            
    return f

def coeffK1(x,y):
    K = 1.0 + 0*(x + y)
    return K

def coeffK2(x,y):
    K = 1 + 0.1*(x + y + (x*y))
    return K

# Configure matplotlib to use LaTeX for text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

LeftX = 0.0
RightX = 10.0
LeftY = 0.0
RightY = 5.0
Nx = 200 # number of intervals in x-direction
Ny = 100 # number of intervals in y-direction
hx = (RightX-LeftX)/Nx # grid step in x-direction
hy = (RightY-LeftY)/Ny # grid step in y-direction

A = FDLaplacian2D(hx, hy, Nx, Ny)
# print(A.toarray())

plt.figure(1)
plt.spy(A, marker='o', markersize= 8, color= 'k')
plt.xlabel('Column Indices')
plt.ylabel('Row Indices')
plt.title("Sparse Negative 2D Laplacian Matrix")

x,y = np.mgrid[(LeftX+hx):(RightX-hx):(Nx-1)*1j, (LeftY+hy):(RightY-hy):(Ny-1)*1j] #change in x is row-wise | change in y is column-wise 

f = sourcefunc(x,y)

# visualizing the source function
plt.ion()
plt.figure(2)
plt.clf()
plt.imshow(f.transpose(), extent = [LeftX+hx, RightX-hx, RightY-hy, LeftY+hy]) 
plt.colorbar(orientation='horizontal')
plt.xlabel('Inner points in X')
plt.ylabel('Inner points in Y')
plt.title("Heat Map of the Source Function")

# lexicographic source vector
fLX = np.reshape(f.transpose(),-1)

u = la.spsolve(A,fLX)

# reshaping the solution vector into 2D array
uArr = np.reshape(u, ((Ny-1),(Nx-1)))

# visualizing the solution
plt.figure(3)
plt.clf()
plt.imshow(uArr, extent = [LeftX+hx, RightX-hx, RightY-hy, LeftY+hy])
plt.colorbar(orientation='horizontal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("FDM solution")

plt.figure(4)
plt.subplot(1,2,1)
plt.imshow(coeffK1(x.transpose(),y.transpose()), extent = [LeftX+hx, RightX-hx, RightY-hy, LeftY+hy])
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.colorbar(orientation='horizontal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Heat map for coefficient function $K = 1$")
plt.subplot(1,2,2)
plt.imshow(coeffK2(x.transpose(),y.transpose()), extent = [LeftX+hx, RightX-hx, RightY-hy, LeftY+hy])
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.colorbar(orientation='horizontal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Heat map for coefficient function $K = x + y + xy$")


A = create2DLFVM(hx, hy, Nx, Ny, coeffK1)
# print(A.toarray())
u = la.spsolve(A,fLX)
uK1 = np.reshape(u, ((Ny-1),(Nx-1)))

A = create2DLFVM(hx, hy, Nx, Ny, coeffK2)
u = la.spsolve(A,fLX)
uK2 = np.reshape(u, ((Ny-1),(Nx-1)))

plt.figure(5)
plt.clf()
plt.imshow(uK1, extent = [LeftX+hx, RightX-hx, LeftY+hy, RightY-hy])
plt.colorbar(orientation='horizontal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("FVM solution for coefficient function $K = 1$")

plt.figure(6)
plt.clf()
plt.imshow(uK2, extent = [LeftX+hx, RightX-hx, LeftY+hy, RightY-hy])
plt.colorbar(orientation='horizontal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("FVM solution for coefficient function $K = x + y + xy$")