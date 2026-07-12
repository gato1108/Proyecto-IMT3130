"""
Sistema de Keller-Segel simplificado (parabolico-eliptico) usando DOLFINx
"""
# Importar las librerías necesarias
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace, Function, form, extract_function_spaces, Constant
from dolfinx.fem.petsc import assemble_matrix, assemble_vector,  create_vector

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter

# Conservación de masa
def calcular_masa(u, domain):
    local_mass = fem.assemble_scalar(form(u*ufl.dx))
    return domain.comm.allreduce(local_mass, op=MPI.SUM)

def correr_simulacion(nx=64, ny=64, k=1.0, l=2.0, T_final=5, dt=0.01, save_every=10, gif=False):
    # Malla y espacio funcional
    # nx, ny definición de la malla
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny,mesh.CellType.triangle) # en base a triángulos
    V = functionspace(domain, ("Lagrange", 1)) # el mismo espacio para u y v

    # Definir los parámetros
    k_diff = Constant(domain, PETSc.ScalarType(k)) # k, coeficiente de difusión
    lam = Constant(domain, PETSc.ScalarType(l)) # lambda, coeficiente de la quimiotaxis
    # paso temporal dt, tiempo final T_final, guardar info cada cierta cantidad de pasos save_every
    n_steps = int(T_final/dt) # pasos
    print(f"Información: k = {k}, lambda = {l}, dt = {dt}, T = {T_final}, pasos = {n_steps}")

    # Trial/test funciones, una para cada ecuación
    u_trial = ufl.TrialFunction(V) # trial para la ecuación de u
    v_trial = ufl.TrialFunction(V) # trial para la ecuación de v
    phi = ufl.TestFunction(V) # test para la ecuación de u
    psi = ufl.TestFunction(V) # test para la ecuación de v

    # Paso actual y nuevo
    u_m = Function(V, name="u_m") # u actual
    v_m = Function(V, name="v_m") # v actual
    u_new = Function(V, name="u") # u_{m+1} nuevo
    v_new = Function(V, name="v") # v_{m+1} nuevo

    # Condición inicial
    def u0_expr(x):
        return 1.0 + 2.0*np.exp(-100.0*((x[0]-0.5)**2+(x[1]-0.5)**2))
    u_m.interpolate(u0_expr) 

    # Definir matrices M, A, B
    M_form = form(ufl.inner(u_trial, phi)*ufl.dx)
    A_form = form((1.0/dt)*ufl.inner(u_trial, phi)*ufl.dx+k_diff*ufl.inner(ufl.grad(u_trial), ufl.grad(phi))*ufl.dx)
    B_form = form(ufl.inner(ufl.grad(v_trial), ufl.grad(psi))*ufl.dx +ufl.inner(v_trial, psi)*ufl.dx)

    A_mat = assemble_matrix(A_form) 
    A_mat.assemble()

    B_mat = assemble_matrix(B_form)
    B_mat.assemble()

    M_mat = assemble_matrix(M_form)
    M_mat.assemble()

    # Solvers PETSc
    solver_u = PETSc.KSP().create(domain.comm)
    solver_u.setOperators(A_mat) # u se resuelve con A
    solver_u.setType(PETSc.KSP.Type.PREONLY)
    solver_u.getPC().setType(PETSc.PC.Type.LU) # por LU

    solver_v = PETSc.KSP().create(domain.comm)
    solver_v.setOperators(B_mat) # v se resuelve con B
    solver_v.setType(PETSc.KSP.Type.PREONLY)
    solver_v.getPC().setType(PETSc.PC.Type.LU) # por LU

    # F(u, v)
    F_form = form(ufl.inner((lam *u_m)*ufl.grad(v_m), ufl.grad(phi))*ufl.dx)

    # Vectores que se calculan en cada iteración
    F_vec = create_vector(extract_function_spaces(F_form))
    b_vec = create_vector(extract_function_spaces(A_form))
    c_vec = create_vector(extract_function_spaces(B_form))

    # Calcular v_0 resolviendo Bv_0 = Mu_0
    M_mat.mult(u_m.x.petsc_vec, c_vec) #c = Mu_0
    solver_v.solve(c_vec, v_m.x.petsc_vec) # B
    v_m.x.scatter_forward()

    # Guardar datos
    frames_data_u = []
    frames_data_v = []
    time_values = []

    # Guardar inicio
    frames_data_u.append(u_m.x.array.copy())
    frames_data_v.append(v_m.x.array.copy())
    time_values.append(0.0)

    area = domain.comm.allreduce(fem.assemble_scalar(form(fem.Constant(domain, PETSc.ScalarType(1.0)) * ufl.dx)), op=MPI.SUM)
    masa = calcular_masa(u_m, domain)
    u_mean = masa/area
    print(f"Masa inicial = {masa:.6f}, Area = {area:.6f}, u_mean (equilibrio esperado) = {u_mean:.6f}")

    # Resolución en el loop
    t = 0.0
    for m in range(n_steps):
        t += dt
        # b = (1/dt)Mu_m+lambdaF(u_m, v_m)
        with F_vec.localForm() as loc:
            loc.set(0.0) # limpia F
        assemble_vector(F_vec, F_form) # calcula F
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        M_mat.mult(u_m.x.petsc_vec, b_vec) # b = Mu_m
        b_vec.scale(1.0/dt) # b = (1/dt)Mu_m
        b_vec.axpy(1.0, F_vec) # b = (1/dt)Mu_m+lambdaF(u_m, v_m)

        # Reolver Au_{m+1}=b
        solver_u.solve(b_vec, u_new.x.petsc_vec)
        u_new.x.scatter_forward()

        # c = M u_{m+1}
        M_mat.mult(u_new.x.petsc_vec, c_vec)

        # Resolver Bv_{m+1} = c
        solver_v.solve(c_vec, v_new.x.petsc_vec)
        v_new.x.scatter_forward()

        # Actualizar: u_{m+1}, v_{m+1}
        u_m.x.array[:] = u_new.x.array
        v_m.x.array[:] = v_new.x.array
        u_m.x.scatter_forward()
        v_m.x.scatter_forward()
        
        # Guarda info para mostar cada save_every pasos
        if (m+1)%save_every == 0:
            frames_data_u.append(u_m.x.array.copy())
            frames_data_v.append(v_m.x.array.copy())
            time_values.append(t)
            masa = calcular_masa(u_m, domain)
            if domain.comm.rank == 0:
                print(f"Paso {m+1}/{n_steps}, t = {t:.4f}, max(u) = {u_m.x.array.max():.4f}, masa = {masa:.4f}")
    
    if gif:
        # Visualización:
        dof_coords = V.tabulate_dof_coordinates()
        x = dof_coords[:, 0]
        y = dof_coords[:, 1]

        num_cells_local = domain.topology.index_map(2).size_local
        cells = V.dofmap.list[:num_cells_local]
        triangulation = mtri.Triangulation(x, y, cells)

        fig = plt.figure(figsize=(11, 5))
        # [left, bottom, width, height] en coordenadas de figura (0 a 1)
        ax_u  = fig.add_axes([0.06, 0.15, 0.36, 0.7])
        cax_u = fig.add_axes([0.44, 0.15, 0.02, 0.7])

        ax_v  = fig.add_axes([0.56, 0.15, 0.36, 0.7])
        cax_v = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        def update(frame_idx):
            ax_u.clear()
            ax_v.clear()
            cax_u.clear()
            cax_v.clear()

            current_u = frames_data_u[frame_idx]
            current_v = frames_data_v[frame_idx]
            current_time = time_values[frame_idx]

            contour_u = ax_u.tricontourf(triangulation, current_u, levels=50, cmap='inferno')
            fig.colorbar(contour_u, cax=cax_u)
            ax_u.set_title(f"u   t={current_time:.3f}  min={current_u.min():.3f} max={current_u.max():.3f}", pad=15.0)
            ax_u.set_xlabel("X")
            ax_u.set_ylabel("Y")
            ax_u.set_aspect("equal")

            contour_v = ax_v.tricontourf(triangulation, current_v, levels=50, cmap='viridis')
            fig.colorbar(contour_v, cax=cax_v)
            ax_v.set_title(f"v   t={current_time:.3f}  min={current_v.min():.3f} max={current_v.max():.3f}", pad=15.0)
            ax_v.set_xlabel("X")
            ax_v.set_ylabel("Y")
            ax_v.set_aspect("equal")

            return contour_u, contour_v

        anim = FuncAnimation(fig, update, frames=len(frames_data_u), interval=50)
        writer = PillowWriter(fps=5)
        anim.save(f"keller_segel_evolution_{nx}_{dt}_{k}_{l}.gif", writer=writer)
        plt.close(fig)
    return u_m, v_m

#correr_simulacion(k=2.0, l=0.0, dt=0.001, save_every=25, gif=True)