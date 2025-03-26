# Solution of 1D Poisson's equation with FDM
# Chirag Bansal (c) 2024

import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib to use LaTeX for text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# function definitions
def func1(x):
    return (3*x) - 2

def func2(x):
    return (x**2) + (3*x) - 2

def fdm(bc, x, N):
    xgrid, h = np.linspace(x[0], x[1], N+1, retstep=True)   #discretising the domain into N intervals
    
    # source functions
    f1 = func1(xgrid)
    f2 = func2(xgrid)
    
    #exact solutions of the poisson equation with different source functions
    u1ex = -((xgrid**3)/2) + (xgrid**2) + ((3/2)*xgrid) + 1
    u2ex = -((xgrid**4)/12) - ((xgrid**3)/2) + (xgrid**2) + ((15/4)*xgrid) + 1
    
    #system matrix
    A = np.diag([(-2)]*(N-1),0)
    k = -1

    while(True):
        B = np.diag([1]*(N-2),k)
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] += B[i][j]

        if(k==-1):
            k = 1
        else:
            break

    A = (-(1/(h**2)))*A

    # source function arrays
    f1rhs = f1[1:N].copy()
    f2rhs = f2[1:N].copy()

    # inserting the boundary conditions to the source function arrays
    f1rhs[0] += bc[0]/(h**2)
    f1rhs[N-2] += bc[1]/(h**2)
    f2rhs[0] += bc[0]/(h**2)
    f2rhs[N-2] += bc[1]/(h**2)

    # solving for the discretised PDE
    u1 = np.linalg.solve(A,f1rhs)
    u2 = np.linalg.solve(A,f2rhs)

    # appending boundary conditions to the FD solution
    u1 = np.append(u1, bc[1])
    u1 = np.insert(u1, 0, bc[0])
    u2 = np.append(u2, bc[1])
    u2 = np.insert(u2, 0, bc[0])

    return xgrid, f1, f2, u1ex, u2ex, A, u1, u2

def globerr(u1, u2, u1ex, u2ex, n): #returns a 1D array corresponding to RMSE of ui (i=1,2)
    err = np.zeros(2)
    err[0] = np.sqrt((1/(n-1)) * np.sum((u1 - u1ex)**2))
    err[1] = np.sqrt((1/(n-1)) * np.sum((u2 - u2ex)**2))

    return err

#----Main----

# domain parameters
bc = [1, 1] #left and right boundary conditions for the domain 
x = [0, 3]  #start and end of the domain

while True: #input error handling
    try:
        n = int(input("Enter number of sub-intervals to create: "))    #number of subintervals needed
        if n >= 2:
            break
        else:
            print("Please enter a sub-interval greater than or equal to 2.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

xgrid, f1, f2, u1ex, u2ex, A, u1, u2 = fdm(bc, x, n)

# eigen values of the Laplacian matrix
print(np.sort(np.linalg.eig(A)[0]))   #get eigen values of the nxn system matrix | index 0 returns only the e-values

# plot for source functions f1 and f2
plt.figure(1)
plt.plot(xgrid,f1, '-o', color='blue', label='$f_1(x)$')
plt.plot(xgrid,f2, '-o', color='red', label='$f_2(x)$')
plt.legend()
plt.xlabel('Grid Values')
plt.ylabel('Source Function')

# plot for exact solution of poisson equation
plt.figure(2)
plt.plot(xgrid,u1ex, '-o', color='blue', label='$u_1^{ex}(x)$')
plt.plot(xgrid,u2ex, '-o', color='red', label='$u_2^{ex}(x)$')
plt.legend()
plt.xlabel('Grid Values')
plt.ylabel('Poisson equation')

# structure of the system matrix
plt.figure(3)
plt.spy(A, marker='o', markersize= 8, color= 'red')
plt.xlabel('Column Indices')
plt.ylabel('Row Indices')

# plot for the numerical and the exact solution
plt.figure(4)
plt.plot(xgrid,u1ex, '-o', color='blue', label='$u_1^{ex}(x)$')
plt.plot(xgrid,u1, '--o', color='blue', label='$u_1(x)$')
plt.plot(xgrid,u2ex, '-o', color='red', label='$u_2^{ex}(x)$')
plt.plot(xgrid,u2, '--o', color='red', label='$u_2(x)$')
plt.legend()
plt.xlabel('Grid Values')
plt.ylabel('u(x)')

# computing the global errors for n=5
err = globerr(u1, u2, u1ex, u2ex, n)
print(err)

# computing global errors for u1 and u2 as a function of n
gerr = [[0 for i in range(2)] for j in range(1)]
for i in range(2,151):
    _, _, _, u1ex, u2ex, _, u1, u2 = fdm(bc, x, i)
    gerr = np.vstack((gerr, globerr(u1, u2, u1ex, u2ex, i))) 

gerr = np.delete(gerr, (0), axis=0)
gerr = np.log(gerr)

# plot for rate of convergence
plt.figure(5)
plt.plot(np.arange(2,len(gerr[:,0])+2),gerr[:,0], '-', color='blue', label='$\epsilon_{u_1(x)}$')
plt.plot(np.arange(2,len(gerr[:,1])+2),gerr[:,1], '-', color='red', label='$\epsilon_{u_2(x)}$')
plt.legend()
plt.xlabel('Number of sub intervals, $N$')
plt.ylabel('$ln(\epsilon)$')
plt.show()