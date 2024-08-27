import numpy as np
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm


# Configuration
urdf_path = "/Users/tommasoandina/Desktop/Doosan_h2515-main/model.urdf"
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
root_link = 'base'
end_effector = 'link6'  

# Initialize KinDynComputations
kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF

# Simulation parameters
T_SIMULATION = 3  # Total simulation time in seconds
dt = 1/16000  # Time step in seconds
N = int(T_SIMULATION / dt)  # Number of time steps

# Initial conditions
q0 = np.zeros(num_dof)
dq0 = np.zeros(num_dof)
H_b = np.eye(4)
v_b = np.zeros(6)

# Optimal control parameters (assuming these are found)
optimal_kp = 1000
optimal_kd = 2 * np.sqrt(optimal_kp)

# Reference position
q_des = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])  # Example desired position

# Dynamics functions
mass_matrix_fun = kinDyn.mass_matrix_fun()
bias_force_fun = kinDyn.bias_force_fun()
jacobian_fun = kinDyn.jacobian_fun(end_effector)

class PDController:
    def __init__(self, kp, kd, q_des):
        self.kp = kp
        self.kd = kd
        self.q_des = q_des

    def control(self, q, dq):
        error = self.q_des - q
        tau = self.kp * error - self.kd * dq
        return tau

# Define external forces as a summation of sinusoids with given amplitudes
def external_forces(t):
    frequencies = np.arange(10, 201, 1)  # Frequencies from 10 to 200 incremented by 1
    amplitudes = np.array([50*2, 60*2, 90*2])  # Amplitudes for x, y, z
    forces = np.zeros(3)
    for i, freq in enumerate(frequencies):
        forces[0] += amplitudes[0] * np.sin(2 * np.pi * freq * t)  # x-axis
        forces[1] += amplitudes[1] * np.sin(2 * np.pi * freq * t)  # y-axis
        forces[2] += amplitudes[2] * np.sin(2 * np.pi * freq * t)  # z-axis
    return forces

# Initialize controller with optimal parameters
controller = PDController(optimal_kp, optimal_kd, q_des)

# Initialize arrays for storing data
q = np.zeros((num_dof, N+1))
dq = np.zeros((num_dof, N+1))
tau = np.zeros((num_dof, N))

q[:, 0] = q0
dq[:, 0] = dq0

# Initialize tracking error
min_tracking_error = float('inf')

for i in range(N):
    t = i * dt

    # Control
    tau[:, i] = controller.control(q[:, i], dq[:, i])

    # Dynamics
    M2 = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
    h2 = bias_force_fun(H_b, q[:, i], v_b, dq[:, i])[6:]
    J_full = jacobian_fun(H_b, q[:, i])[0:3, 6:]

    # External forces
    f_ext = external_forces(t)
    
    # RK4 integration
    def acceleration(dq_current):
        # Converti il risultato in un vettore 1D di forma (6,)
        return np.linalg.solve(M2, tau[:, i] - h2 + J_full.T @ f_ext - M2 @ dq_current).reshape(-1)

    k1 = dt * acceleration(dq[:, i])
    k2 = dt * acceleration(dq[:, i] + 0.5 * k1)
    k3 = dt * acceleration(dq[:, i] + 0.5 * k2)
    k4 = dt * acceleration(dq[:, i] + k3)

    # Aggiorna dq e q
    dq[:, i+1] = dq[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    q[:, i+1] = q[:, i] + dt * dq[:, i]


# Calculate tracking error
tracking_error = np.linalg.norm(q_des - q[:, -1])

if tracking_error < min_tracking_error:
    min_tracking_error = tracking_error


# Plotting the best result
plt.figure(figsize=(12, 8))
for j in range(num_dof):
    plt.subplot(num_dof, 1, j+1)
    plt.plot(np.arange(N+1)*dt, q[j, :], label=f'Joint {j+1}')
    plt.plot(np.arange(N+1)*dt, np.full(N+1, q_des[j]), '--', label=f'Desired q_{j+1}')
    plt.ylabel(f'q_{j+1} [rad]')
    plt.legend()
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()

print(f"Optimal kp: {optimal_kp} with tracking error {min_tracking_error}")
print("Simulation completed.")
