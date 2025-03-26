# FEM for 1D Poisson equation with mixed inhomogeneous BC's
# works in python3 and donfinx version 0.9.0

# Neil Budko (c) 2024

import numpy as np
import matplotlib.pyplot as plt

import ufl
from ufl import ds, dx, grad, inner
from basix.ufl import element
from dolfinx import fem, mesh, __version__

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem

dolfinx_version = '0.9.0'

if __version__ != dolfinx_version:
    print('dolfinx version is',__version__,', this code requires version',dolfinx_version)
else:
    print('dolfinx version',dolfinx_version,'found, solving...')

dpi=100

# input data
x0 = 0 # start of interval, Dirichlet BC boundary
Lx = 1 # length of interval
uDL = 3 # Dirichlet BC value
uNR = 5 # Neumann BC value (rest of boundary)

nx = 10 # number of vertices in the mesh

print('-----------------------------------------------------------------------')

# domain with mesh
domain = mesh.create_interval(comm=MPI.COMM_WORLD,points=(x0,x0+Lx),nx=nx,)
# coordinates of vertices
dim = domain.geometry.dim
grid = domain.geometry.x[:,0:dim]
# domain properties
print('domain.geometry.x\n',domain.geometry.x)
print('domain.geometry.dim',domain.geometry.dim)
print('domain.geometry.x[:,0:dim]\n',domain.geometry.x[:,0:dim])
# Linear Lagrange finite element
P1 = element("Lagrange",cell=domain.topology.cell_name(),degree=1)
# Scalar function space V with scalar elements P1
V = fem.functionspace(domain, P1)
# mesh adjacency list
print('V.dofmap.list\n',V.dofmap.list)

# imposing Dirichlet BC on the function space
dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x0))
bc_D = fem.dirichletbc(value=ScalarType(uDL), dofs=dofs_L, V=V)

print('value of the Dirichlet BC =',bc_D.g.value)
print('value of the Neumann BC =',uNR)

# source function (set as zero)
f = fem.Constant(domain,ScalarType(0.0))

# trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# structure of the bilinear form a(u,v)
a = inner(grad(u),grad(v))*dx
# structure of the linear form L(v)
L = f*v*dx - ScalarType(uNR)*v*ds

# defining unknown function and FEM forms
u = fem.Function(V)
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# fixing the value of the source function
f_val = 0.0 # magnitude of the constant source function
f.value = np.copy(f_val)
# setting up linear problem
problem = LinearProblem(a, L, bcs=[bc_D], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# solving the linear problem
u = problem.solve()

# inspecting assembled A and b, and the computed solution u
Ad = problem.A.convert("dense") # for small matrices only!
Aarr = Ad.getDenseArray()
barr = np.copy(problem.b.array)
uarr = np.copy(u.x.array)
print('assembled matrix A\n',Aarr)
print('source function f =',f_val)
print('assembled vector b\n',barr)
print('computed solution vector u\n',u.x.array)

# changing the value of the source function
f_val = 20.0 # magnitude of the constant source function
f.value = np.copy(f_val)
# resetting linear problem
problem = LinearProblem(a, L, bcs=[bc_D], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# solving the linear problem
u = problem.solve()

# inspecting assembled b and the computed solution u
barr1 = np.copy(problem.b.array)
uarr1 = np.copy(u.x.array)
print('source function f =',f_val)
print('assembled vector b\n',barr1)
print('computed solution vector u\n',u.x.array)

# plotting with Matplotlib
# plt.ion()
plt.figure(1,dpi=dpi)
# plt.clf()
fig1, ax = plt.subplots(nrows=1, ncols=1, num=1)
gridDof = V.tabulate_dof_coordinates()[:,dim-1]
ax.plot(gridDof,uarr,'-o',color='tab:blue',label=r'FEM solution for $f(x) = 0$')
ax.plot(gridDof,uarr1,'-o',color='tab:orange',label=r'FEM solution for $f(x) = 20$')
ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x)$')
print('...done')
plt.show()