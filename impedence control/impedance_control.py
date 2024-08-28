import numpy as np
from numpy import nan
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time
import sys
import pickle
from scipy.interpolate import CubicSpline
from scipy.linalg import eigvals

# Configuration
urdf_path = "/Users/tommasoandina/Desktop/Doosan_h2515-main/model.urdf"
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
root_link = 'base'
end_effector = 'link6'

# Initialize KinDynComputations
kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF

# Dynamics functions
mass_matrix_fun = kinDyn.mass_matrix_fun()
bias_force_fun = kinDyn.bias_force_fun()
jacobian_fun = kinDyn.jacobian_fun(end_effector)
forward_kinematics_fun = kinDyn.forward_kinematics_fun(end_effector)

# Simulation parameters
conf = {
    'T_SIMULATION': 8.0,
    'dt_control': 1/1000,   # Time step for controller: 1 ms
    'dt_simulation': 1/16000,  # Time step for simulator: 1/16 ms
    'PRINT_T': 1.0,
    'simulate_real_time': False,
    'kp_j': 100.0,
    'kd_j': 20.0,
    'kp': 1000.0,  # Fixed kp for the controller
    'kd': 2 * np.sqrt(1000),  # Corresponding kd for critical damping
    'q0': np.array([1.50882228e-04, 1.02054791e-02, 1.84429995e+00, 0.0, 0.0, 0.0])  # Initial position
}

# Placeholder for simulator
class Simulator:
    def __init__(self):
        self.q = np.zeros(num_dof)
        self.v = np.zeros(num_dof)
        self.tau_c = np.zeros(num_dof)
        self.dv = np.zeros(num_dof)

    def init(self, q0):
        self.q = q0
        self.v = np.zeros_like(q0)

    def simulate(self, tau, dt, ndt):
        def dynamics(q, v, tau):
            H_b = np.eye(4)
            v_b = np.zeros(6)
            M2 = mass_matrix_fun(H_b, q)[6:, 6:]
            h2 = bias_force_fun(H_b, q, v_b, v)[6:]
            ddq = np.linalg.inv(M2) @ (tau - h2)
            return ddq

        for _ in range(ndt):
            k1_v = dynamics(self.q, self.v, tau) * dt
            k1_q = self.v * dt

            q2 = self.q + 0.5 * k1_q
            v2 = self.v + 0.5 * k1_v
            k2_v = dynamics(q2, v2, tau) * dt
            k2_q = v2 * dt

            q3 = self.q + 0.5 * k2_q
            v3 = self.v + 0.5 * k2_v
            k3_v = dynamics(q3, v3, tau) * dt
            k3_q = v3 * dt

            q4 = self.q + k3_q
            v4 = self.v + k3_v
            k4_v = dynamics(q4, v4, tau) * dt
            k4_q = v4 * dt

            self.v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
            self.q += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6

            if np.any(np.isnan(self.q)) or np.any(np.isnan(self.v)):
                raise ValueError("NaN detected in simulator state after integration")

            self.q = np.clip(self.q, -1e3, 1e3)
            self.v = np.clip(self.v, -1e3, 1e3)

simu = Simulator()

def generate_trajectory(t):
    t_points = [0.0, 2.0, 5.0, 8.0]  # Updated to match the simulation duration
    x_points = [
        [1.50882228e-04, 1.02054791e-02, 1.84429995e+00],  # Start point
        [0.5, 0.7, 0.6],  # Intermediate control point
        [1.0, 1.5, 1.2],  # Another intermediate control point
        [1.5, 1.3, 1.0]   # End point
    ]

    # Ensure continuity by using cubic spline interpolation
    x_spline = CubicSpline(t_points, [p[0] for p in x_points], bc_type='natural')
    y_spline = CubicSpline(t_points, [p[1] for p in x_points], bc_type='natural')
    z_spline = CubicSpline(t_points, [p[2] for p in x_points], bc_type='natural')

    x_ref = np.array([x_spline(t), y_spline(t), z_spline(t)])
    dx_ref = np.array([x_spline(t, 1), y_spline(t, 1), z_spline(t, 1)])
    ddx_ref = np.array([x_spline(t, 2), y_spline(t, 2), z_spline(t, 2)])

    return x_ref, dx_ref, ddx_ref

def check_stability(M2, h2, J_full):
    try:
        M2 = np.array(M2).astype(float)
        h2 = np.array(h2).astype(float)
        J_full = np.array(J_full).astype(float)
        
        num_dof = M2.shape[0]

        if h2.ndim == 1:
            h2 = h2.reshape(-1, 1)

        A = np.block([
            [np.zeros((num_dof, num_dof)), np.eye(num_dof)],
            [-np.linalg.inv(M2) @ J_full.T @ J_full, -np.linalg.inv(M2) @ np.eye(num_dof)]
        ])

        eigenvalues = eigvals(A)
        
        if np.all(np.real(eigenvalues) < 0):
            print("Il sistema è stabile.")
        else:
            print("Il sistema potrebbe essere instabile.")
    except Exception as e:
        print(f"Error in check_stability: {e}")

tests = [
    {'controller': 'OSC'},
    {'controller': 'IC'}
]

# Simulazione principale con time step diversi
kp = conf['kp']
kd = conf['kd']

tracking_err_osc = []
tracking_err_ic = []

for (test_id, test) in enumerate(tests):
    description = str(test_id) + ' Controller ' + test['controller'] + ' kp=1000'
    print(description)
    simu.init(conf['q0'])

    nx, ndx = 3, 3
    N_control = int(conf['T_SIMULATION'] / conf['dt_control'])
    N_simulation = int(conf['dt_control'] / conf['dt_simulation'])
    tau = np.empty((num_dof, N_control)) * nan
    tau_c = np.empty((num_dof, N_control)) * nan
    q = np.empty((num_dof, N_control + 1)) * nan
    v = np.empty((num_dof, N_control + 1)) * nan
    dv = np.empty((num_dof, N_control + 1)) * nan
    x = np.empty((nx, N_control)) * nan
    dx = np.empty((ndx, N_control)) * nan
    ddx = np.empty((ndx, N_control)) * nan
    x_ref = np.empty((nx, N_control)) * nan
    dx_ref = np.empty((ndx, N_control)) * nan
    ddx_ref = np.empty((ndx, N_control)) * nan
    ddx_des = np.empty((ndx, N_control)) * nan

    t = 0.0
    PRINT_N = int(conf['PRINT_T'] / conf['dt_control'])

    for i in range(0, N_control):
        time_start = time.time()

        x_ref[:, i], dx_ref[:, i], ddx_ref[:, i] = generate_trajectory(t)

        v[:, i] = np.squeeze(simu.v)
        q[:, i] = np.squeeze(simu.q)

        if i % PRINT_N == 0:  # Adjusted print frequency
            print(f"Step {i}:")
            print(f"q = {q[:, i]}")
            print(f"v = {v[:, i]}")
        
        try:
            H_b = np.eye(4)
            v_b = np.zeros(6)
            dq = v[:, i]

            M2 = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
            h2 = bias_force_fun(H_b, q[:, i], v_b, dq)[6:]
            J_full = jacobian_fun(H_b, q[:, i])[0:3, 6:]

            end_effector_position = np.array(forward_kinematics_fun(H_b, q[:, i]))[:3, 3]
            end_effector_velocity = J_full @ dq

            dJdq = np.zeros(3)
            delta = 1e-6
            for j in range(num_dof):
                q_plus = q[:, i].copy()
                q_minus = q[:, i].copy()
                q_plus[j] += delta
                q_minus[j] -= delta
                J_plus = jacobian_fun(H_b, q_plus)[0:3, 6:]
                J_minus = jacobian_fun(H_b, q_minus)[0:3, 6:]
                dJdq += (J_plus - J_minus) @ dq / (2 * delta)

            x[:, i] = end_effector_position
            dx[:, i] = np.array(end_effector_velocity).flatten()

            ddx_fb = kp * (x_ref[:, i] - x[:, i]) + kd * (dx_ref[:, i] - dx[:, i])
            ddx_des[:, i] = ddx_ref[:, i] + ddx_fb

            Minv = cs.inv(M2)
            J_Minv = J_full @ Minv
            Lambda = cs.inv(J_Minv @ J_full.T)

            mu = Lambda @ (J_Minv @ h2 - dJdq)
            f = Lambda @ ddx_des[:, i] + mu

            J_T_pinv = Lambda @ J_Minv
            NJ = np.eye(num_dof) - J_full.T @ J_T_pinv
            tau_0 = M2 @ (conf['kp_j'] * (conf['q0'] - q[:, i]) - conf['kd_j'] * v[:, i])

            if test['controller'] == 'OSC':
                tau[:, i] = np.squeeze(J_full.T @ f) + np.squeeze(NJ @ (tau_0 + h2))
            elif test['controller'] == 'IC':
                tau[:, i] = np.squeeze(h2 + J_full.T @ (8 * ddx_fb) + NJ @ tau_0)
            else:
                print('ERROR: Unknown controller', test['controller'])
                sys.exit(0)

            # Simulazione con il passo del simulatore
            simu.simulate(tau[:, i], conf['dt_simulation'], N_simulation)
            tau_c[:, i] = simu.tau_c
            dv[:, i] = simu.dv
            t += conf['dt_control']

            if i % PRINT_N == 0:
                print(f"Step {i}:")
                print(f"q = {q[:, i]}")
                print(f"v = {v[:, i]}")
                print(f"x = {x[:, i]}")
                print(f"dx = {dx[:, i]}")
                print(f"ddx = {ddx[:, i]}")
                print()

            time_spent = time.time() - time_start
            if conf['simulate_real_time'] and time_spent < conf['dt_control']:
                time.sleep(conf['dt_control'] - time_spent)

        except Exception as e:
            print(f"Error at step {i}: {e}")
            print(f"q = {q[:, i]}")
            print(f"v = {v[:, i]}")
            print(f"x = {x[:, i]}")
            print(f"dx = {dx[:, i]}")
            print(f"ddx = {ddx[:, i]}")
            break

    # Calcola l'errore di tracking usando la norma all'infinito
    tracking_err = np.max(np.linalg.norm(x_ref - x, axis=0, ord=np.inf))
    desc = test['controller'] + ' kp=1000'
    if test['controller'] == 'OSC':
        tracking_err_osc.append({'value': tracking_err, 'description': desc})
    elif test['controller'] == 'IC':
        tracking_err_ic.append({'value': tracking_err, 'description': desc})
    else:
        print('ERROR: Unknown controller', test['controller'])

    print('Max tracking error (norma all\'infinito): %.3f m\n' % (tracking_err))

    # Analisi di stabilità
    check_stability(M2, h2, J_full)

# Plot delle Traiettorie dei Giunti
(f, ax) = plt.subplots(num_dof, 1, figsize=(10, 20))
tt = np.arange(0.0, N_control * conf['dt_control'], conf['dt_control'])

for i in range(num_dof):
    ax[i].plot(tt, q[i, :-1], label=f'q_{i}')
    ax[i].set_xlabel('Time [s]')
    ax[i].set_ylabel(f'q_{i} [rad]')
    ax[i].legend()
plt.suptitle(f'Traiettorie dei Giunti per kp = {conf["kp"]}')
plt.show()

# Plot delle Traiettorie Cartesiane
plt.figure()
for i in range(nx):
    plt.plot(tt, x[i, :], label=f'x_{i}')
    plt.plot(tt, x_ref[i, :], '--', label=f'x_ref_{i}')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title(f'Traiettorie Cartesiane per kp = {conf["kp"]}')
plt.legend()
plt.show()
