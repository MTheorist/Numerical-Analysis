# FEM for 2D convection-diffusion equation 
# compact Gaussian sources and spatiallly nonuniform wind
# runs on Python 3
# DOLFINx version: 0.9.0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata

import ufl
from basix.ufl import element
from dolfinx import fem, mesh, geometry
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from petsc4py import PETSc
from dolfinx.fem.petsc import ( assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc )

from dolfinx import __version__

dolfinx_version = '0.9.0'

if __version__ != dolfinx_version:
    print('dolfinx version is',__version__,', this code requires version',dolfinx_version)
else:
    print('dolfinx version',dolfinx_version,'found')


x0 = -3 # domain corner, x coordinate
y0 = -3 # domain corner, y coordinate
Lx = 6 # domain extent in x-direction
Ly = 6 # domain extent in y-direction
T = 5 # time interval
As = 5 # source magnitude
alpha = 10 # source decay rate in space
B = 1.5 # wind constant
D = 0.0025 # diffusivity constant
bcValue = 0 # Dirichlet BC value


nx = 51 # number of cells in x-direction
ny = 51 # number of cells in y-direction

nt = 10 # number of time steps
dt = T/nt # time step

# creating domain and mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array(x0,y0), np.array(x0+Lx,y0+Ly)], [nx,ny], mesh.CellType.triangle)

# Scalar function space V with scalar elements P1
P1 = element("Lagrange",domain.topology.cell_name(),1)
V = fem.functionspace(domain, P1)
# Vector function space W with vector elements P2
P2 = element("Lagrange",domain.topology.cell_name(),1,shape=(2,))
W = fem.functionspace(domain, P2)

# Create Dirichlet boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(bcValue), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# source function class
class source_function():
    def __init__(self):
        self.t = 0
        self.alpha = 0
        self.A = 0
    def eval(self, x):
        f = A*(np.exp(-alpha((x[0]-(-2.4))**2+((x[1])**2)))+np.exp(-alpha((x[0])**2+(x[1]-1)**2)))
        return f
# velocity x-component function class
class velocity_x:
    def __init__(self):
        self.t = 0
        self.B = 0
    def eval(self, x):
        wx = B - ((x[1])/(np.sqrt(x[0]**2+x[1]**2)))
        return wx
# velocity y-component function class
class velocity_y:
    def __init__(self):
        self.t = 0
    def eval(self, x):
        wy = ((x[0])/(np.sqrt(x[0]**2+x[1]**2)))
        return wy
# initial condition Python function
def initial_condition(x):
    u = x[0]*0
    return u

def get_value(fun,x,domain):
    '''This function computes the value of the solution at a chosen point
    fun - FEM funciton, such as the current iteration u_k
    x - numpy 3D array with the point coordinates, e.g., x = np.array([0.5,0,0])
    domain - dolfinx mesh object
    '''
    bb_tree = geometry.bb_tree(domain,domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree,x=x)
    colliding_cells = geometry.compute_colliding_cells(domain,cell_candidates,x=x)
    value = fun.eval(x=x,cells=colliding_cells.array[0])
    return value[0]

# source fem function
fun = source_function()
fun.alpha = alpha
fun.A = As
f = fem.Function(V)
f.interpolate(fun.eval)

# velocity fem vector function
vxf = velocity_x()
vxf.B = B
vyf = velocity_y()
vx = fem.Function(V)
vx.interpolate(vxf.eval)
vy = fem.Function(V)
vy.interpolate(vyf.eval)
w = fem.Function(W)
w.sub(0).interpolate(vx)
w.sub(1).interpolate(vy)

#------------------------------------------------------------------------------
# visualize velocity vector field w
# ....

# initial condition fem function
u_k = fem.Function(V)
u_k.name = "u_k"
u_k.interpolate(initial_condition)

# solution variable fem function
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(lambda x: x[0]*0)

# Weak form: a(u,v) = L(v) for backward Euler method
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = (u*v*ufl.dx) + (dt * D * ufl.dot(ufl.grad(u),ufl.grad(v)) * ufl.dx) + (dt * ufl.dot(w,grad(u)))# bi-linear form
L = (u_k + dt * f) * v * ufl.dx # linear form
u = fem.Function(V)
bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

t = 0.0

# preparing to plot solution at each time step
plt.ion()
plt.figure(1)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
unow = u_k.x.array
dof_coordinates = V.tabulate_dof_coordinates()
dof_x = dof_coordinates[:,0]
dof_y = dof_coordinates[:,1]
im =  ax.tripcolor(dof_x,dof_y,unow,shading='gouraud')
ax.set_aspect('equal', adjustable='box')
cbar = fig.colorbar(im,ax=ax,orientation='horizontal')
ax.set_title(r'$u(t)$, t = '+str(np.round(t,2)))


# Time stepping
for k in range(nt):
    t = t + dt # updating time

    # update the right hand side re-using the initial vector

    # apply Dirichlet boundary condition to the vector

    # solve linear system
    solver.solve(...)
    uh.x.scatter_forward()

    # overwrite previous time step

    # plot solution snapshot
    unow = u_k.x.array
    im.remove()
    im = ax.tripcolor(dof_x,dof_y,unow,shading='gouraud')
    cbar.update_normal(im)
    ax.set_title(r'$u(t)$, t = '+str(np.round(t,2)))
    plt.pause(0.2)

# print solution value at a point

