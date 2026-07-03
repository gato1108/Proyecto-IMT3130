from mpi4py import MPI
from petsc4py import PETSc

import time
import shutil
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import ufl

from basix.ufl import element, mixed_element
from dolfinx import fem, mesh, plot, io
from dolfinx.fem import petsc as fem_petsc
from dolfinx.fem.petsc import LinearProblem
from scipy import sparse
from ufl import dx, grad, dot
import pyvista as pv
from dolfinx import plot
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import dolfinx_mpc
except ImportError:
    dolfinx_mpc = None

comm = MPI.COMM_WORLD
rank = comm.rank

# Crear Dominio y espacios de funciones
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
V_main = fem.functionspace(domain, ("Lagrange", 1)) 
V_aux  = fem.functionspace(domain, ("Lagrange", 1)) 

# Funciones para almacenar iteracones
u_h = fem.Function(V_main)    
u_n = fem.Function(V_main)    
phi_h = fem.Function(V_aux)   

# Parámetros
dt = fem.Constant(domain, 0.01)       
k = fem.Constant(domain, 0.005)       
lmbda = fem.Constant(domain, 2.0)    

# Formulación auxiliar
phi = ufl.TrialFunction(V_aux)
psi = ufl.TestFunction(V_aux)

f_u = u_n 

a_aux = dot(grad(phi), grad(psi)) * dx + phi * psi * dx
L_aux = f_u * psi * dx

# Problema auxiliar
prob_secondary = LinearProblem(a_aux, L_aux, u=phi_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"},  petsc_options_prefix="basic_linear_problem")

# Funciones centrales
u = ufl.TrialFunction(V_main)
chi = ufl.TestFunction(V_main)

# Operadores
b_h = -u * dot(grad(phi_h), grad(chi)) * dx 
a_main = (1.0 / dt) * u * chi * dx + k * dot(grad(u), grad(chi)) * dx + lmbda * b_h
L_main = (1.0 / dt) * u_n * chi * dx

# Problema central
prob_main = LinearProblem(a_main, L_main, u=u_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"},  petsc_options_prefix="basic_linear_problem")

# 5.1 Aleatorio
#u_n.interpolate(lambda x: 1.0 + 0.1*np.random.rand(x.shape[1])) 

# 5.2 Equilibrada
#u_n.interpolate(lambda x: 1.0 + 0.05*np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1]))

# 5.3 Gaussiana
#u_n.interpolate(lambda x: 1.0 + np.exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2) / 0.04))

# 5.4 Varios puntos
#u_n.interpolate(lambda x: 1.0 + 0.05*np.cos(8*np.pi*x[0])*np.cos(8*np.pi*x[1]))

# 5.5 Puntas
#u_n.interpolate(lambda x:1.0+0.15*np.exp(-((x[0]-0.25)**2+(x[1]-0.25)**2)/0.005)+0.15*np.exp(-((x[0]-0.75)**2+(x[1]-0.25)**2)/0.005)+0.15*np.exp(-((x[0]-0.25)**2+(x[1]-0.75)**2)/0.005)+0.15*np.exp(-((x[0]-0.75)**2+(x[1]-0.75)**2)/0.005))

# 5.6 franjas
#u_n.interpolate(lambda x:1.0 + 0.1*np.cos(6*np.pi*x[1]))

# 5.7 tablero
u_n.interpolate(lambda x: 1.0 +0.1*np.cos(8*np.pi*x[0])*np.cos(8*np.pi*x[1]))

# 5.8 espiral
#u_n.interpolate(lambda x: 1.0+0.1*np.cos(12*np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)+4*np.arctan2(x[1]-0.5,x[0]-0.5)))

u_h.x.array[:] = u_n.x.array[:]

# Guardar 
vtx_u = io.VTXWriter(domain.comm, "main_sol_u.bp", [u_h])
vtx_phi = io.VTXWriter(domain.comm, "secondary_sol_phi.bp", [phi_h])

vtx_u.write(0.0)
vtx_phi.write(0.0)

# Flujo temporal
t = 0.0
T_final = 3.0
step = 0

while t < T_final:
    t += float(dt)
    step += 1
    
    # Problema auxiliar y central
    prob_secondary.solve()
    prob_main.solve()
    
    # Guarda
    vtx_u.write(t)
    vtx_phi.write(t)

    # Actualizar funciones
    u_n.x.array[:] = u_h.x.array[:]
    
    if step % 10 == 0:
        print(f"Step {step} completed | Time = {t:.2f}")

vtx_u.close()
vtx_phi.close()


# Visualizar
plotter = pv.Plotter(title="Keller-Segel Solution Visualization")
topology, cell_types, geometry = plot.vtk_mesh(V_main)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["u_sol"] = u_h.x.array.real
warped_grid = grid.warp_by_scalar(scalars="u_sol", factor=0.1)
plotter.add_mesh(warped_grid, cmap="viridis", show_edges=True)
plotter.show()


import matplotlib.pyplot as plt

# Extract coordinates and solution values
points = V_main.tabulate_dof_coordinates()[:, :2] # Get X and Y coordinates
x, y = points[:, 0], points[:, 1]
values = u_h.x.array.real

# Use matplotlib's triangulation mapping to handle finite element nodes
plt.figure(figsize=(6, 5))
plt.tricontourf(x, y, values, levels=50, cmap='inferno')
plt.colorbar(label='Cell Density')
plt.title('Keller-Segel Aggregation')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('keller_segel_snapshot.png', dpi=300)
plt.show()

plt.ion() 
fig, ax = plt.subplots(figsize=(6, 5))
points = V_main.tabulate_dof_coordinates()[:, :2]
x, y = points[:, 0], points[:, 1]

# Condiciones iniciales

# 5.1 Aleatorio
#u_n.interpolate(lambda x: 1.0 + 0.1*np.random.rand(x.shape[1])) 

# 5.2 Equilibrada
#u_n.interpolate(lambda x: 1.0 + 0.05*np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1]))

# 5.3 Gaussiana
#u_n.interpolate(lambda x: 1.0 + np.exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2) / 0.04))

# 5.4 Varios puntos
#u_n.interpolate(lambda x: 1.0 + 0.05*np.cos(8*np.pi*x[0])*np.cos(8*np.pi*x[1]))

# 5.5 Puntas
#u_n.interpolate(lambda x:1.0+0.15*np.exp(-((x[0]-0.25)**2+(x[1]-0.25)**2)/0.005)+0.15*np.exp(-((x[0]-0.75)**2+(x[1]-0.25)**2)/0.005)+0.15*np.exp(-((x[0]-0.25)**2+(x[1]-0.75)**2)/0.005)+0.15*np.exp(-((x[0]-0.75)**2+(x[1]-0.75)**2)/0.005))

# 5.6 franjas
#u_n.interpolate(lambda x:1.0 + 0.1*np.cos(6*np.pi*x[1]))

# 5.7 tablero
u_n.interpolate(lambda x: 1.0 +0.1*np.cos(8*np.pi*x[0])*np.cos(8*np.pi*x[1]))

# 5.8 espiraal
#u_n.interpolate(lambda x: 1.0+0.1*np.cos(12*np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)+4*np.arctan2(x[1]-0.5,x[0]-0.5)))


u_h.x.array[:] = u_n.x.array[:]


t = 0.0
T_final = 3.0
step = 0

# Computar gif
while t < T_final:
    t += float(dt)
    step += 1
    
    # Resolver primer paso
    prob_secondary.solve()
    prob_main.solve()
    
    # Alternar pasos
    if step % 2 == 0: 
        ax.clear()    
        values = u_h.x.array.real
        contour = ax.tricontourf(x, y, values, levels=50, cmap='inferno')
        
        ax.set_title(f"Cell Density Evolution (t = {t:.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        if step == 2:
            cbar = fig.colorbar(contour, ax=ax, label='Density')
        
        plt.draw()
        plt.pause(0.001) 
    u_n.x.array[:] = u_h.x.array[:]

plt.ioff() 
plt.show() 

# Almacenar gif
fig, ax = plt.subplots(figsize=(6, 5))
points = V_main.tabulate_dof_coordinates()[:, :2]
x, y = points[:, 0], points[:, 1]

# 5.1 Aleatorio
#u_n.interpolate(lambda x: 1.0 + 0.1*np.random.rand(x.shape[1])) 

# 5.2 Equilibrada
#u_n.interpolate(lambda x: 1.0 + 0.05*np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1]))

# 5.3 Gaussiana
#u_n.interpolate(lambda x: 1.0 + np.exp(-((x[0]-0.5)**2 + (x[1]-0.5)**2) / 0.04))

# 5.4 Varios puntos
#u_n.interpolate(lambda x: 1.0 + 0.05*np.cos(8*np.pi*x[0])*np.cos(8*np.pi*x[1]))

# 5.5 Puntas geniales
#u_n.interpolate(lambda x:1.0+0.15*np.exp(-((x[0]-0.25)**2+(x[1]-0.25)**2)/0.005)+0.15*np.exp(-((x[0]-0.75)**2+(x[1]-0.25)**2)/0.005)+0.15*np.exp(-((x[0]-0.25)**2+(x[1]-0.75)**2)/0.005)+0.15*np.exp(-((x[0]-0.75)**2+(x[1]-0.75)**2)/0.005))

# 5.6 franjas
#u_n.interpolate(lambda x:1.0 + 0.1*np.cos(6*np.pi*x[1]))

# 5.7 tablero
u_n.interpolate(lambda x: 1.0 +0.1*np.cos(8*np.pi*x[0])*np.cos(8*np.pi*x[1]))

# 5.8 espiraal
#u_n.interpolate(lambda x: 1.0+0.1*np.cos(12*np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)+4*np.arctan2(x[1]-0.5,x[0]-0.5)))

u_h.x.array[:] = u_n.x.array[:]

frames_data = [] 
time_values = [] 

t = 0.0
T_final = 3.0

while t < T_final:
    t += float(dt)
    prob_secondary.solve()
    prob_main.solve()
    
    frames_data.append(np.array(u_h.x.array.real))
    time_values.append(t)
    
    u_n.x.array[:] = u_h.x.array[:]

def update(frame_idx):
    ax.clear()
    current_values = frames_data[frame_idx]
    current_time = time_values[frame_idx]
    contour = ax.tricontourf(x, y, current_values, levels=50, cmap='inferno')
    ax.set_title(f"Cell Density Evolution (t = {current_time:.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return contour


anim = FuncAnimation(fig, update, frames=len(frames_data), interval=50)


writer = PillowWriter(fps=20)
anim.save("keller_segel_evolution.gif", writer=writer)
