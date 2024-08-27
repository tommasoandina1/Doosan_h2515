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
    'dt': 1/16000, 
    'PRINT_T': 1.0,
    'simulate_real_time': False,  
    'kp_j': 100.0,
    'kd_j': 20.0,
    'q0': np.zeros(num_dof)
}

# Placeholder for tests
tests = [
    {'controller': 'OSC'},
    {'controller': 'IC'}
]

# Placeholder for simulator
class Simulator:
    def __init__(self):
        self.q = np.zeros(num_dof)
        self.v = np.zeros(num_dof)
        self.tau_c = np.zeros(num_dof)
        self.dv = np.zeros(num_dof)

    def init(self, q0):
        self.q = q0
        self.v = np.zeros_like(q0)  # Initialize velocities to zero

    def simulate(self, tau, dt, ndt):
        # Runge-Kutta 4th order integration for simulation purposes
        def dynamics(q, v, tau):
            dv = tau  # Assuming direct proportionality for simplicity
            return dv

        for _ in range(ndt):
            k1_v = dynamics(self.q, self.v, tau) * dt
            k1_q = self.v * dt

            k2_v = dynamics(self.q + 0.5 * k1_q, self.v + 0.5 * k1_v, tau) * dt
            k2_q = (self.v + 0.5 * k1_v) * dt

            k3_v = dynamics(self.q + 0.5 * k2_q, self.v + 0.5 * k2_v, tau) * dt
            k3_q = (self.v + 0.5 * k2_v) * dt

            k4_v = dynamics(self.q + k3_q, self.v + k3_v, tau) * dt
            k4_q = (self.v + k3_v) * dt

            self.v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
            self.q += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6

            # Check for NaN values after integration
            if np.any(np.isnan(self.q)) or np.any(np.isnan(self.v)):
                raise ValueError("NaN detected in simulator state after integration")

            # Limit the values of q and v to avoid instability
            self.q = np.clip(self.q, -1e3, 1e3)
            self.v = np.clip(self.v, -1e3, 1e3)

simu = Simulator()

# Function to generate the reference trajectory using cubic splines
def generate_trajectory(t):
    # Define key points for the trajectory
    t_points = [0.0, 2.0, 5.0, 8.0]
    x_points = [
        [0.0, 0.0, 0.0],  # at t=0.0
        [1.0, 1.0, 1.0],  # at t=2.0
        [1.0, 2.0, 0.9],  # at t=5.0
        [1.0, 1.0, 0.9]   # at t=8.0
    ]
    
    # Create cubic splines for position, velocity, and acceleration
    x_spline = CubicSpline(t_points, [p[0] for p in x_points], bc_type='clamped')
    y_spline = CubicSpline(t_points, [p[1] for p in x_points], bc_type='clamped')
    z_spline = CubicSpline(t_points, [p[2] for p in x_points], bc_type='clamped')
    
    # Compute reference position, velocity, and acceleration
    x_ref = np.array([x_spline(t), y_spline(t), z_spline(t)])
    dx_ref = np.array([x_spline(t, 1), y_spline(t, 1), z_spline(t, 1)])
    ddx_ref = np.array([x_spline(t, 2), y_spline(t, 2), z_spline(t, 2)])
    
    return x_ref, dx_ref, ddx_ref

# Main simulation loop
tracking_err_osc = []
tracking_err_ic = []

for (test_id, test) in enumerate(tests):
    description = str(test_id) + ' Controller ' + test['controller'] + ' kp=1000'
    print(description)
    kp = 1000  # kp is fixed at 1000
    kd = 2 * np.sqrt(kp)
    simu.init(conf['q0'])

    nx, ndx = 3, 3
    N = int(conf['T_SIMULATION'] / conf['dt'])
    tau = np.empty((num_dof, N)) * nan
    tau_c = np.empty((num_dof, N)) * nan
    q = np.empty((num_dof, N + 1)) * nan
    v = np.empty((num_dof, N + 1)) * nan
    dv = np.empty((num_dof, N + 1)) * nan
    x = np.empty((nx, N)) * nan
    dx = np.empty((ndx, N)) * nan
    ddx = np.empty((ndx, N)) * nan
    x_ref = np.empty((nx, N)) * nan
    dx_ref = np.empty((ndx, N)) * nan
    ddx_ref = np.empty((ndx, N)) * nan
    ddx_des = np.empty((ndx, N)) * nan

    t = 0.0
    PRINT_N = int(conf['PRINT_T'] / conf['dt'])

    for i in range(0, N):
        time_start = time.time()

        # set reference trajectory using sinusoidal function
        x_ref[:, i], dx_ref[:, i], ddx_ref[:, i] = generate_trajectory(t)

        # read current state from simulator
        v[:, i] = simu.v
        q[:, i] = simu.q

        # Check for NaN values in initial state
        if np.any(np.isnan(q[:, i])) or np.any(np.isnan(v[:, i])):
            raise ValueError(f"NaN detected in initial state at step {i}")

        # Debug prints for current state
        if i % 1000 == 0:
            print(f"Step {i}:")
            print(f"q = {q[:, i]}")
            print(f"v = {v[:, i]}")
        
        try:
            # compute mass matrix M, bias terms h
            H_b = np.eye(4)  # Assuming H_b is the identity matrix for simplicity
            v_b = np.zeros(6)  # Assuming v_b is zero for simplicity
            dq = v[:, i]  # Using current velocity

            # Debug prints for inputs to mass_matrix_fun
            if i % 1000 == 0:
                print(f"H_b = \n{H_b}")
                print(f"q[:, i] = {q[:, i]}")

            M2 = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
            h2 = bias_force_fun(H_b, q[:, i], v_b, dq)[6:]
            J_full = jacobian_fun(H_b, q[:, i])[0:3, 6:]

            # Debug print per la matrice di massa
            if i % 1000 == 0:
                print(f"M2 = \n{M2}")

            # External forces
            end_effector_position = np.array(forward_kinematics_fun(H_b, q[:, i]))[:3, 3]
            end_effector_velocity = J_full @ dq

            # Check for NaN values
            if np.any(np.isnan(end_effector_position)) or np.any(np.isnan(end_effector_velocity)):
                raise ValueError(f"NaN detected in end_effector_position or end_effector_velocity at step {i}")

            # Compute dJdq manually
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

            # Compute current end-effector position and velocity
            x[:, i] = end_effector_position
            dx[:, i] = np.array(end_effector_velocity).flatten()  # Convert to NumPy array and flatten

            # Check for NaN values
            if np.any(np.isnan(x[:, i])) or np.any(np.isnan(dx[:, i])):
                raise ValueError(f"NaN detected in x or dx at step {i}")

            # implement your control law here
            ddx_fb = kp * (x_ref[:, i] - x[:, i]) + kd * (dx_ref[:, i] - dx[:, i])
            
            # Debug print for feedback acceleration
            if i % 1000 == 0:
                print(f"ddx_fb = {ddx_fb}")

            # Check for NaN values in feedback acceleration
            if np.any(np.isnan(ddx_fb)):
                raise ValueError(f"NaN detected in ddx_fb at step {i}")

            ddx_des[:, i] = ddx_ref[:, i] + ddx_fb

            # Check for invalid values before proceeding
            if np.any(np.isnan(ddx_des[:, i])):
                raise ValueError(f"Invalid ddx_des at step {i}: {ddx_des[:, i]}")

            Minv = cs.inv(M2)
            J_Minv = J_full @ Minv
            Lambda = cs.inv(J_Minv @ J_full.T)
            
            # Debug print for Lambda
            if i % 1000 == 0:
                print(f"Lambda = {Lambda}")

            mu = Lambda @ (J_Minv @ h2 - dJdq)
            
            # Debug print for mu
            if i % 1000 == 0:
                print(f"mu = {mu}")

            f = Lambda @ ddx_des[:, i] + mu
            
            # Debug print for f
            if i % 1000 == 0:
                print(f"f = {f}")

            # Check for NaN values in Lambda, mu, and f
            if np.any(np.isnan(Lambda)) or np.any(np.isnan(mu)) or np.any(np.isnan(f)):
                raise ValueError(f"NaN detected in Lambda, mu, or f at step {i}")

            # secondary task
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

            # send joint torques to simulator
            simu.simulate(tau[:, i], conf['dt'], 1)
            tau_c[:, i] = simu.tau_c
            dv[:, i] = simu.dv
            t += conf['dt']

            # Check for NaN values after simulation step
            if np.any(np.isnan(simu.q)) or np.any(np.isnan(simu.v)):
                raise ValueError(f"NaN detected in simulator state after simulation step {i}")

            # Debug prints
            if i % 1000 == 0:  # Print every 1000 steps for more frequent debugging
                print(f"Step {i}:")
                print(f"q = {q[:, i]}")
                print(f"v = {v[:, i]}")
                print(f"x = {x[:, i]}")
                print(f"dx = {dx[:, i]}")
                print(f"ddx = {ddx[:, i]}")
                print()

            time_spent = time.time() - time_start
            if conf['simulate_real_time'] and time_spent < conf['dt']:
                time.sleep(conf['dt'] - time_spent)

        except Exception as e:
            print(f"Error at step {i}: {e}")
            print(f"q = {q[:, i]}")
            print(f"v = {v[:, i]}")
            print(f"x = {x[:, i]}")
            print(f"dx = {dx[:, i]}")
            print(f"ddx = {ddx[:, i]}")
            break
    
    tracking_err = np.sum(norm(x_ref - x, axis=0)) / N
    desc = test['controller'] + ' kp=1000'
    if test['controller'] == 'OSC':
        tracking_err_osc.append({'value': tracking_err, 'description': desc})
    elif test['controller'] == 'IC':
        tracking_err_ic.append({'value': tracking_err, 'description': desc})
    else:
        print('ERROR: Unknown controller', test['controller'])

    print('Average tracking error %.3f m\n' % (tracking_err))


2
# Plot joint trajectories
(f, ax) = plt.subplots(num_dof, 1, figsize=(10, 20))
tt = np.arange(0.0, N * conf['dt'], conf['dt'])
for i in range(num_dof):
    ax[i].plot(tt, q[i, :-1], label=f'q_{i}')
    ax[i].set_xlabel('Time [s]')
    ax[i].set_ylabel(f'q_{i} [rad]')
    ax[i].legend()
plt.show()

# Plot Cartesian trajectories
plt.figure()
for i in range(nx):
    plt.plot(tt, x[i, :], label=f'x_{i}')
    plt.plot(tt, x_ref[i, :], '--', label=f'x_ref_{i}')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.show()

# Save the values of q to a text file
np.savetxt('/Users/tommasoandina/Desktop/q_values.txt', q)
