# Importar las librerías necesarias
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, Function, form, extract_function_spaces, Constant
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter


def correr_simulacion(nx=128, ny=128, Lx=1.0, Ly=1.0, k_valor=0.1, lam=3.2, a=1.0, b=1.0, c=1.0, u_centro=1.0, ruido_amplitud=0.02, 
                      T_final=40, dt=0.001, save_every=10, gif=False):
    # Malla y espacio funcional
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    V = functionspace(domain, ("Lagrange", 1)) # mismo espacio para u y v

    # Definir los parametros
    k = Constant(domain, PETSc.ScalarType(k_valor)) # k, difusion
    l = Constant(domain, PETSc.ScalarType(lam)) # lambda, quimiotaxis
    a_valor = Constant(domain, PETSc.ScalarType(a)) # a, crecimiento logístico
    b_valor = Constant(domain, PETSc.ScalarType(b)) # b, disminución de v
    c_valor = Constant(domain, PETSc.ScalarType(c)) # c, producción de v

    n_steps = int(T_final/dt)
    print(f"k = {k_valor}, lambda = {lam}, a = {a}, b = {b}, c = {c}, dt = {dt}, T = {T_final}, pasos = {n_steps}")
    print(f"Equilibrio: u = 1, v = {c/b:.4f}")

    # Trial/test funciones, una para cada ecuacion
    u_trial = ufl.TrialFunction(V)
    v_trial = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)
    psi = ufl.TestFunction(V)

    # Paso actual y nuevo
    u_m = Function(V, name="u_m")
    v_m = Function(V, name="v_m")
    u_new = Function(V, name="u")
    v_new = Function(V, name="v")

    # Condicion inicial
    rng = np.random.default_rng(0) # semilla para replicar
    def u0_expr(x):
        ruido = ruido_amplitud*u_centro*(2.0*rng.random(x.shape[1])-1.0)
        return u_centro+ruido
    u_m.interpolate(u0_expr)

    # Definir matrices M, A, B
    M_form = form(ufl.inner(u_trial, phi)*ufl.dx)
    A_form = form((1.0/dt)*ufl.inner(u_trial, phi)*ufl.dx+k*ufl.inner(ufl.grad(u_trial), ufl.grad(phi))*ufl.dx)
    B_form = form(ufl.inner(ufl.grad(v_trial), ufl.grad(psi))*ufl.dx+ b_valor*ufl.inner(v_trial, psi)*ufl.dx)

    A_mat = assemble_matrix(A_form)
    A_mat.assemble()

    B_mat = assemble_matrix(B_form)
    B_mat.assemble()

    M_mat = assemble_matrix(M_form)
    M_mat.assemble()

    # Solvers PETSc
    solver_u = PETSc.KSP().create(domain.comm)
    solver_u.setOperators(A_mat)
    solver_u.setType(PETSc.KSP.Type.PREONLY)
    solver_u.getPC().setType(PETSc.PC.Type.LU)

    solver_v = PETSc.KSP().create(domain.comm)
    solver_v.setOperators(B_mat)
    solver_v.setType(PETSc.KSP.Type.PREONLY)
    solver_v.getPC().setType(PETSc.PC.Type.LU)

    # F(u, v)
    F_form = form(ufl.inner((l*u_m)*ufl.grad(v_m), ufl.grad(phi))*ufl.dx)

    # G(u)= a*u*(1-u)
    G_form = form(a_valor*ufl.inner(u_m*(1.0-u_m), phi)*ufl.dx)

    # Vectores que se calculan en cada iteracion
    F_vec = create_vector(extract_function_spaces(F_form))
    G_vec = create_vector(extract_function_spaces(G_form))
    b_vec = create_vector(extract_function_spaces(A_form))
    c_vec = create_vector(extract_function_spaces(B_form))

    # Calcular v_0 resolviendo B*v_0 = c*M*u_0
    M_mat.mult(u_m.x.petsc_vec, c_vec)
    c_vec.scale(c_valor)
    solver_v.solve(c_vec, v_m.x.petsc_vec)
    v_m.x.scatter_forward()

    # Guardar datos
    frames_data_u = []
    frames_data_v = []
    time_values = []

    # Guardar inicio
    frames_data_u.append(u_m.x.array.copy())
    frames_data_v.append(v_m.x.array.copy())
    time_values.append(0.0)

    # Resolucion en el loop
    t = 0.0
    for m in range(n_steps):
        t += dt
        # Termino de quimiotaxis
        with F_vec.localForm() as loc:
            loc.set(0.0) # limpia F
        assemble_vector(F_vec, F_form) # calcula F
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # Termino logistico
        with G_vec.localForm() as loc:
            loc.set(0.0) # limpia G
        assemble_vector(G_vec, G_form) # Calcula G
        G_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # b = (1/dt)*M*u_m + F(u_m,v_m) + G(u_m)
        M_mat.mult(u_m.x.petsc_vec, b_vec)
        b_vec.scale(1.0/dt) #*(1/dt)
        b_vec.axpy(1.0, F_vec) #+ F(u_m,v_m)
        b_vec.axpy(1.0, G_vec) #+ G(u_m)

        # Resolver A*u_{m+1} = b
        solver_u.solve(b_vec, u_new.x.petsc_vec)
        u_new.x.scatter_forward()

        # c_vec = c*M*u_{m+1}
        M_mat.mult(u_new.x.petsc_vec, c_vec)
        c_vec.scale(c)

        # Resolver B*v_{m+1} = c
        solver_v.solve(c_vec, v_new.x.petsc_vec)
        v_new.x.scatter_forward()

        # Actualizar
        u_m.x.array[:] = u_new.x.array
        v_m.x.array[:] = v_new.x.array
        u_m.x.scatter_forward()
        v_m.x.scatter_forward()

        if (m + 1) % save_every == 0:
            frames_data_u.append(u_m.x.array.copy())
            frames_data_v.append(v_m.x.array.copy())
            time_values.append(t)
            if domain.comm.rank == 0:
                print(f"paso {m+1}/{n_steps}, t = {t:.4f}, max(u) = {u_m.x.array.max():.4f}")

    if gif:
        # Visualizacion
        dof_coords = V.tabulate_dof_coordinates()
        x = dof_coords[:, 0]
        y = dof_coords[:, 1]

        num_cells_local = domain.topology.index_map(2).size_local
        cells = V.dofmap.list[:num_cells_local]
        triangulation = mtri.Triangulation(x, y, cells)

        fig = plt.figure(figsize=(14, 5))
        ax_u  = fig.add_axes([0.055, 0.15, 0.36, 0.7])
        cax_u = fig.add_axes([0.43, 0.15, 0.02, 0.7])

        ax_v  = fig.add_axes([0.545, 0.15, 0.36, 0.7])
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
        anim.save(f"pattern_keller_segel_logistico_{nx}_{dt}_{k}_{lam}_{a}_{b}_{c}.gif", writer=writer)
        plt.close(fig)


correr_simulacion(k_valor=0.0625, lam=21.0, a=7.0, b=32.0, c=1.0, Lx=6.0, Ly=3.4641, nx = 192, ny=111, dt=0.001, T_final=70,
                  u_centro=1.0, ruido_amplitud=0.05, gif=True, save_every=1000)