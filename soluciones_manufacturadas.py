"""
Soluciones manufacturadas para Keller Segel
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, Function, form, extract_function_spaces, Constant
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import matplotlib.pyplot as plt

# Solución exacta manufacturada
def u_exacta_np(x, y, t):
    return t*np.cos(np.pi*x)*np.cos(np.pi*y)

def v_exacta_np(x, y, t):
    return (1+t)*np.cos(np.pi*x)*np.cos(np.pi*y)

def u_exacta_ufl(x, t):
    return t*ufl.cos(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1])
 
def v_exacta_ufl(x, t):
    return (1+t)*ufl.cos(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1])

def f_u(x, y, t, k, lam):
    p1 = np.cos(np.pi*x)*np.cos(np.pi*y)*(1+2*k*t*np.pi**2) 
    p2 = 2*(np.cos(np.pi*y)**2)*(np.cos(np.pi*x)**2)
    p3 = (np.cos(np.pi*y)**2)*(np.sin(np.pi*x)**2)
    p4 = (np.cos(np.pi*x)**2)*(np.sin(np.pi*y)**2)
    return p1-lam*t*(1+t)*(np.pi**2)*(p2-p3-p4) 
    
def f_v(x, y, t):
    return np.cos(np.pi*x)*np.cos(np.pi*y)*(1+2*(1+t)*np.pi**2)


# Error L2 y H1
def L2_error(u_func, u_ex_ufl, quad_degree=8):
    dx = ufl.dx(metadata={"quadrature_degree": quad_degree})
    diff = u_func - u_ex_ufl
    local_val = fem.assemble_scalar(form(diff**2*dx))
    return np.sqrt(u_func.function_space.mesh.comm.allreduce(local_val, op=MPI.SUM))
 
def H1_error(u_func, u_ex_ufl, quad_degree=8):
    dx = ufl.dx(metadata={"quadrature_degree": quad_degree})
    diff = u_func - u_ex_ufl
    error_form = form((diff**2+ufl.inner(ufl.grad(diff), ufl.grad(diff)))*dx)
    local_val = fem.assemble_scalar(error_form)
    return np.sqrt(u_func.function_space.mesh.comm.allreduce(local_val, op=MPI.SUM))

def correr_simulacion_m(nx=64, ny=64, k=1.0, lam=2.0, T_final=3.0, dt=0.001):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
    V = functionspace(domain, ("Lagrange", 1))

    k_diff = Constant(domain, PETSc.ScalarType(k))
    lam_const = Constant(domain, PETSc.ScalarType(lam))

    u_trial = ufl.TrialFunction(V)
    v_trial = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)
    psi = ufl.TestFunction(V)

    u_m = Function(V, name="u_m")
    v_m = Function(V, name="v_m")
    u_new = Function(V, name="u")
    v_new = Function(V, name="v")

    # condicion inicial: sol exacta en t=0
    u_m.interpolate(lambda x: u_exacta_np(x[0], x[1], 0.0))

    M_form = form(ufl.inner(u_trial, phi)*ufl.dx)
    A_form = form((1.0/dt)*ufl.inner(u_trial, phi)*ufl.dx+k_diff*ufl.inner(ufl.grad(u_trial), ufl.grad(phi))*ufl.dx)
    B_form = form(ufl.inner(ufl.grad(v_trial), ufl.grad(psi))*ufl.dx +ufl.inner(v_trial, psi)*ufl.dx)

    A_mat = assemble_matrix(A_form)
    A_mat.assemble()
    B_mat = assemble_matrix(B_form)
    B_mat.assemble()
    M_mat = assemble_matrix(M_form)
    M_mat.assemble()

    solver_u = PETSc.KSP().create(domain.comm)
    solver_u.setOperators(A_mat)
    solver_u.setType(PETSc.KSP.Type.PREONLY)
    solver_u.getPC().setType(PETSc.PC.Type.LU)

    solver_v = PETSc.KSP().create(domain.comm)
    solver_v.setOperators(B_mat)
    solver_v.setType(PETSc.KSP.Type.PREONLY)
    solver_v.getPC().setType(PETSc.PC.Type.LU)

    # termino no lineal (igual que antes)
    F_form = form(lam_const*ufl.inner(u_m*ufl.grad(v_m), ufl.grad(phi))*ufl.dx)

    # términos extra para la solución manufacturada
    f_u_expr = Function(V, name="f_u") # de la ecuación de u
    f_v_expr = Function(V, name="f_v") # de la ecuación de v

    f_u_form = form(ufl.inner(f_u_expr, phi)*ufl.dx)
    f_v_form = form(ufl.inner(f_v_expr, psi)*ufl.dx)

    F_vec = create_vector(extract_function_spaces(F_form))
    b_vec = create_vector(extract_function_spaces(A_form))
    c_vec = create_vector(extract_function_spaces(B_form))
    fu_vec = create_vector(extract_function_spaces(f_u_form))
    fv_vec = create_vector(extract_function_spaces(f_v_form))

    # Resolver B v0 = M u0 + f_v(t_0) 
    f_v_expr.interpolate(lambda xx: f_v(xx[0], xx[1], 0.0, k, lam))
    with fv_vec.localForm() as loc:
        loc.set(0.0)
    assemble_vector(fv_vec, f_v_form)
    fv_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    M_mat.mult(u_m.x.petsc_vec, c_vec)
    c_vec.axpy(1.0, fv_vec)
    solver_v.solve(c_vec, v_m.x.petsc_vec)
    v_m.x.scatter_forward()

    n_steps = int(round(T_final/dt))
    t = 0.0
    for m in range(n_steps):
        t += dt
        with F_vec.localForm() as loc:
            loc.set(0.0)
        assemble_vector(F_vec, F_form)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        #fu
        f_u_expr.interpolate(lambda x: f_u(x[0], x[1], t, k, lam))
        with fu_vec.localForm() as loc:
            loc.set(0.0)
        assemble_vector(fu_vec, f_u_form)
        fu_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        M_mat.mult(u_m.x.petsc_vec, b_vec)
        b_vec.scale(1.0/dt)
        b_vec.axpy(1.0, F_vec)
        b_vec.axpy(1.0, fu_vec) #fu

        solver_u.solve(b_vec, u_new.x.petsc_vec)
        u_new.x.scatter_forward()

        # f_v evaluada en t_{m+1}
        f_v_expr.interpolate(lambda x: f_v(x[0], x[1], t))
        with fv_vec.localForm() as loc:
            loc.set(0.0)
        assemble_vector(fv_vec, f_v_form)
        fv_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        M_mat.mult(u_new.x.petsc_vec, c_vec)
        c_vec.axpy(1.0, fv_vec) #fv

        solver_v.solve(c_vec, v_new.x.petsc_vec)
        v_new.x.scatter_forward()

        u_m.x.array[:] = u_new.x.array
        v_m.x.array[:] = v_new.x.array
        u_m.x.scatter_forward()
        v_m.x.scatter_forward()

    # Error con sol exacta
    x = ufl.SpatialCoordinate(domain)
    u_ex_ufl = u_exacta_ufl(x, T_final)
    v_ex_ufl = v_exacta_ufl(x, T_final)
 
    err_L2_u = L2_error(u_m, u_ex_ufl)
    err_H1_u = H1_error(u_m, u_ex_ufl)
    err_L2_v = L2_error(v_m, v_ex_ufl)
    err_H1_v = H1_error(v_m, v_ex_ufl)

    print(f"nx={nx}, h={1.0/nx:.5f}, dt={dt:.5f}, error_L2 u ={err_L2_u:.6e}, error_H1 u ={err_H1_u:.6e}, error_L2 v ={err_L2_v:.6e}, error_H1 v ={err_H1_v:.6e}")
    return err_L2_u, err_H1_u, err_L2_v, err_H1_v


k_val = 1.0
lam_val = 3.0
T_final = 0.8
dt = 0.0001 #0.0005 
nx_list = [8, 16, 32, 64, 128]

h_values = []
L2_values_u = []
H1_values_u = []
L2_values_v = []
H1_values_v = []

print("Convergencia:")
for nx in nx_list:
    eL2u, eH1u, eL2v, eH1v = correr_simulacion_m(nx=nx, ny=nx, k=k_val, lam=lam_val, T_final=T_final, dt=dt)
    h_values.append(1.0/nx)
    L2_values_u.append(eL2u)
    H1_values_u.append(eH1u)
    L2_values_v.append(eL2v)
    H1_values_v.append(eH1v)

h_values = np.array(h_values)
L2_values_u = np.array(L2_values_u)
H1_values_u = np.array(H1_values_u)
L2_values_v = np.array(L2_values_v)
H1_values_v = np.array(H1_values_v)

# L2 u
ref = L2_values_u[0]*(h_values/h_values[0])**2 
plt.figure(figsize=(6, 5))
plt.loglog(h_values, L2_values_u, 'o', label='Error')
plt.loglog(h_values, ref, 'k--', alpha=0.5, label='orden 2')
plt.xlabel("h")
plt.ylabel(r"Error $L^2$")
plt.title("Convergencia $L^2$ de $u$")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_L2_u.png", dpi=150)
plt.close()

# H1 u 
ref = H1_values_u[0]*(h_values/h_values[0])**1
plt.figure(figsize=(6, 5))
plt.loglog(h_values, H1_values_u, 'o', label='Error')
plt.loglog(h_values, ref, 'k--', alpha=0.5, label='orden 1')
plt.xlabel("h")
plt.ylabel(r"Error $H^1$")
plt.title("Convergencia $H^1$ de $u$")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_H1_u.png", dpi=150)
plt.close()

# L2 v
ref = L2_values_v[0]*(h_values/h_values[0])**2 
plt.figure(figsize=(6, 5))
plt.loglog(h_values, L2_values_v, 'o', label='Error')
plt.loglog(h_values, ref, 'k--', alpha=0.5, label='orden 2')
plt.xlabel("h")
plt.ylabel(r"Error $L^2$")
plt.title("Convergencia $L^2$ de $v$")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_L2_v.png", dpi=150)
plt.close()

# H1 v
ref = H1_values_v[0]*(h_values/h_values[0])**1
plt.figure(figsize=(6, 5))
plt.loglog(h_values, H1_values_v, 'o', label='Error')
plt.loglog(h_values, ref, 'k--', alpha=0.5, label='orden 1')
plt.xlabel("h")
plt.ylabel(r"Error $H^1$")
plt.title("Convergencia $H^1$ de $v$")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_H1_v.png", dpi=150)
plt.close()