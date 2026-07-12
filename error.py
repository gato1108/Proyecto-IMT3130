from solver import correr_simulacion #importar el solver

# Importar las librerías necesarias
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import mesh, fem
from dolfinx.fem import Function, form

import matplotlib.pyplot as plt

def calcular_error_L2_referencia(u, u_ref):
    # Malla gruesa y fina de ref
    V = u.function_space
    domain = V.mesh
    V_ref = u_ref.function_space

    # Interpola la referencia en la malla gruesa
    num_cells_local = domain.topology.index_map(2).size_local
    celdas = np.arange(num_cells_local, dtype=np.int32)
    interpolation_data = fem.create_interpolation_data(V, V_ref, celdas)
    u_ref_interpolada = Function(V)
    u_ref_interpolada.interpolate_nonmatching(u_ref, celdas, interpolation_data)

    # Calcula diferencia en malla gruesa
    diff = u-u_ref_interpolada
    local_err = fem.assemble_scalar(form(diff**2*ufl.dx))
    return np.sqrt(domain.comm.allreduce(local_err, op=MPI.SUM))

# Error en el espacio
# Referencia
l = 2.0
k = 1.0
u_ref, v_ref = correr_simulacion(nx=256, ny=256, l=l, k=k, T_final=0.5, dt=0.001, gif=False)

# Iterar para diferentes mallas
nx_list = [8, 16, 32, 64]
h_values = []
errores_u = []
errores_v = []

for nx in nx_list:
    u_h, v_h = correr_simulacion(nx=nx, ny=nx, l=l, k=k, T_final=0.5, dt=0.001, gif=False)
    err_u = calcular_error_L2_referencia(u_h, u_ref)
    err_v = calcular_error_L2_referencia(v_h, v_ref)

    h_values.append(1.0/nx)
    errores_u.append(err_u)
    errores_v.append(err_v)

    print(f"nx={nx}, h={1.0/nx:.5f}, error_L2_u={err_u:.6e}, error_L2_v={err_v:.6e}")

h_values = np.array(h_values)
errores_u = np.array(errores_u)
errores_v = np.array(errores_v)

# Convergencia u
ref = errores_u[0]*(h_values/h_values[0])**2
plt.figure(figsize=(6, 5))
plt.loglog(h_values, errores_u, 'o-', label='Error $u$')
plt.loglog(h_values, ref, 'k--', alpha=0.5, label='orden 2')
plt.xlabel("h")
plt.ylabel(r"Error $L^2$")
plt.title("Convergencia de $u$ vs $u$ referencia")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_u_referencia.png", dpi=150)
plt.close()

# Convergencia v
ref = errores_v[0]*(h_values/h_values[0])**2
plt.figure(figsize=(6, 5))
plt.loglog(h_values, errores_v, 'o-', label='Error $v$')
plt.loglog(h_values, ref, 'k--', alpha=0.5, label='orden 2')
plt.xlabel("h")
plt.ylabel(r"Error $L^2$")
plt.title("Convergencia de $v$ vs $v$ referencia")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_v_referencia.png", dpi=150)
plt.close()



# Error en el tiempo
# Referencia
nx = 128 
dt = 0.0002
u_ref_t, v_ref_t = correr_simulacion(nx=nx, ny=nx, T_final=0.5, dt=dt, gif=False)

# Iterar para diferentes dt
dt_list = [0.02, 0.01, 0.005, 0.0025]
dt_values = []
errores_u_dt = []
errores_v_dt = []

for dt in dt_list:
    u_h, v_h = correr_simulacion(nx=nx, ny=nx, T_final=0.5, dt=dt, gif=False)
    err_u = calcular_error_L2_referencia(u_h, u_ref_t)
    err_v = calcular_error_L2_referencia(v_h, v_ref_t)

    dt_values.append(dt)
    errores_u_dt.append(err_u)
    errores_v_dt.append(err_v)

    print(f"dt={dt}, error_L2_u={err_u:.6e}, error_L2_v={err_v:.6e}")

dt_values = np.array(dt_values)
errores_u_dt = np.array(errores_u_dt)
errores_v_dt = np.array(errores_v_dt)


# Convergencia u
ref = errores_u_dt[0]*(dt_values/dt_values[0])**1  # orden 1
plt.figure(figsize=(6, 5))
plt.loglog(dt_values, errores_u_dt, 'o-', label='Error $u$')
plt.loglog(dt_values, ref, 'k--', alpha=0.5, label='orden 1')
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"Error $L^2$")
plt.title("Convergencia temporal de $u$")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_temporal_u.png", dpi=150)
plt.close()


# Convergencia v
ref = errores_v_dt[0]*(dt_values/dt_values[0])**1   # orden 1
plt.figure(figsize=(6, 5))
plt.loglog(dt_values, errores_v_dt, 'o-', label='Error $v$')
plt.loglog(dt_values, ref, 'k--', alpha=0.5, label='orden 1')
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"Error $L^2$")
plt.title("Convergencia temporal de $v$")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("convergencia_temporal_v.png", dpi=150)
plt.close()