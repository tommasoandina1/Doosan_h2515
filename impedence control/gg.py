import numpy as np
from numpy import nan
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm, pinv
import time as py_time
import sys

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
    'dt': 1/1000,
    'PRINT_T': 1.0,
    'simulate_real_time': False,
    'kp_j': 100.0,
    'kd_j': 20.0,
    'q0': np.zeros(num_dof)
}

# Define tests
tests = [
    {'controller': 'OSC', 'kp': 100},
    {'controller': 'IC', 'kp': 100},
    {'controller': 'OSC', 'kp': 500},
    {'controller': 'IC', 'kp': 500},
    {'controller': 'OSC', 'kp': 1000},
    {'controller': 'IC', 'kp': 1000}
]

class Simulator:
    def __init__(self):
        self.q = np.zeros(num_dof)
        self.v = np.zeros(num_dof)

    def init(self, q0):
        self.q = q0
        self.v = np.zeros_like(q0)

    def simulate(self, tau, dt):
        H_b = np.eye(4)
        M = mass_matrix_fun(H_b, self.q)[6:, 6:]
        h = bias_force_fun(H_b, self.q, np.zeros(6), self.v)[6:]
        
        self.v += dt * np.linalg.solve(M, tau.reshape((-1, 1)) - h.reshape((h.shape[0], 1))).flatten()
        self.q += dt * self.v

def generate_trajectory(t):
    # Define parameters for sinusoidal trajectories
    A = np.array([0.5, 0.5, 0.5])  # Amplitudes
    f = np.array([0.2, 0.3, 0.4])  # Frequencies
    phi = np.array([0, np.pi/3, np.pi/6])  # Phase shifts
    offset = np.array([0.5, 0.5, 0.5])  # Offsets

    # Compute reference position, velocity, and acceleration
    x_ref = A * np.sin(2*np.pi*f*t + phi) + offset
    dx_ref = 2*np.pi*f * A * np.cos(2*np.pi*f*t + phi)
    ddx_ref = -(2*np.pi*f)**2 * A * np.sin(2*np.pi*f*t + phi)

    return x_ref, dx_ref, ddx_ref

# Main simulation loop
for test in tests:
    print(f"Running test: {test['controller']} with kp={test['kp']}")
    
    kp = test['kp']
    kd = 2 * np.sqrt(kp)
    
    N = int(conf['T_SIMULATION'] / conf['dt'])
    tau = np.zeros((num_dof, N))
    q = np.zeros((num_dof, N+1))
    v = np.zeros((num_dof, N+1))
    x = np.zeros((3, N))
    dx = np.zeros((3, N))
    
    simu = Simulator()
    simu.init(conf['q0'])
    t = 0.0
    
    for i in range(N):
        time_start = py_time.time()
        
        # Get current state
        q[:, i] = simu.q
        v[:, i] = simu.v
        
        # Compute desired trajectory
        x_des, dx_des, ddx_des = generate_trajectory(t)
        
        # Compute current end-effector position and velocity
        H_b = np.eye(4)
        H = forward_kinematics_fun(H_b, q[:, i])
        x[:, i] = np.array(H[:3, 3]).flatten()
        J = jacobian_fun(H_b, q[:, i])[:3, 6:]
        dx[:, i] = np.array(J @ v[:, i]).flatten()
        
        # Compute control
        if test['controller'] == 'OSC':
            ddx = ddx_des + kp * (x_des - x[:, i]) + kd * (dx_des - dx[:, i])
            M = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
            h = bias_force_fun(H_b, q[:, i], np.zeros(6), v[:, i])[6:]
            tau[:, i] = np.array(J.T @ ddx + h).flatten()
        elif test['controller'] == 'IC':
            ddq = pinv(J) @ (ddx_des + kp * (x_des - x[:, i]) + kd * (dx_des - dx[:, i]))
            M = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
            h = bias_force_fun(H_b, q[:, i], np.zeros(6), v[:, i])[6:]
            tau[:, i] = np.array(M @ ddq + h).flatten()
        
        # Secondary task (joint space PD control to preferred configuration)
        N = np.eye(num_dof) - J.T @ pinv(J @ J.T) @ J
        tau_0 = conf['kp_j'] * (conf['q0'] - q[:, i]) - conf['kd_j'] * v[:, i]
        tau[:, i] += np.array(N @ tau_0).flatten()
        
        # Send joint torques to simulator
        simu.simulate(tau[:, i], conf['dt'])
        
        t += conf['dt']
        
        # Optional: add real-time simulation
        if conf['simulate_real_time']:
            time_spent = py_time.time() - time_start
            if time_spent < conf['dt']:
                py_time.sleep(conf['dt'] - time_spent)
    
    # Compute and print average tracking error
    error = np.mean(norm(x_des.reshape(3, 1) - x, axis=0))

    print(f"Average tracking error: {error:.5f}")

    # Plot results
    time_array = np.arange(0, conf['T_SIMULATION'], conf['dt'])
    plt.figure(figsize=(12, 8))
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(time_array, x[j, :], label=f'Actual {["x", "y", "z"][j]}')
        plt.plot(time_array, [generate_trajectory(t)[0][j] for t in time_array], '--', label=f'Desired {["x", "y", "z"][j]}')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
    plt.suptitle(f"{test['controller']} - kp={test['kp']}")
    plt.tight_layout()
    plt.show()
