import numpy as np
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Percorso del file contenente i valori di q
file_path = '/Users/tommasoandina/Desktop/q_values.txt'
q_des = np.loadtxt(file_path)

# Configuration
urdf_path = "/Users/tommasoandina/Desktop/Doosan_h2515-main/model.urdf"
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
root_link = 'base'
end_effector = 'link6'  

# Initialize KinDynComputations
kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF


def get_q_des_current(index):
    return q_des[:, index]


# Simulation parameters
T_SIMULATION = 8  # Total simulation time in seconds
dt = 1/16000  # Time step in seconds
N = 128001  # Number of time steps

# Initial conditions
q0 = np.zeros(num_dof)
dq0 = np.zeros(num_dof)
H_b = np.eye(4)
v_b = np.zeros(6)

# Optimal control parameters (assuming these are found)
optimal_kp = 1000
optimal_kd = 2 * np.sqrt(optimal_kp)


# Dynamics functions
mass_matrix_fun = kinDyn.mass_matrix_fun()
bias_force_fun = kinDyn.bias_force_fun()
jacobian_fun = kinDyn.jacobian_fun(end_effector)
forward_kinematics_fun = kinDyn.forward_kinematics_fun(end_effector)

class PDController:
    def __init__(self, kp, kd, q_des):
        self.kp = kp
        self.kd = kd
        self.q_des = q_des

    def control(self, q, dq):
        error = self.q_des - q
        tau = self.kp * error - self.kd * dq
        return tau

# Define the surface

surface_height = 1

def compute_contact_force(position, velocity, k_friction):
    k_contact = 3e4  # Contact stiffness
    d_contact = np.sqrt(k_contact)  # Contact damping
    mu = 0.3

    if position[2] < surface_height:
        force_z = k_contact * (surface_height - position[2]) - d_contact * velocity[2]
        tangential_velocity = velocity[:2]
        tangential_speed = np.linalg.norm(tangential_velocity)
        if tangential_speed > 1e-6:
            friction_direction = -tangential_velocity / tangential_speed
            friction_magnitude = mu * abs(force_z) * k_friction * np.sign(tangential_velocity)
            friction_force = friction_magnitude * friction_direction
        else:
            friction_force = np.zeros(2)
        return np.array([friction_force[0], friction_force[1], force_z])
    else:
        return np.zeros(3)


# Define external forces as a summation of sinusoids with given amplitudes
def external_forces(t):
    frequencies = np.arange(10, 201, 1)  # Frequencies from 10 to 200 incremented by 1
    amplitudes = np.array([100, 120, 180])  # Amplitudes for x, y, z
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
error = float('inf')

def simulation(k_friction):
    velocities = []
    positions = []
    end_effector_velocities = []
    friction_forces = []  
    ee_positions = []
    ee_velocities = []

    def acceleration(dq_current, q_current, tau_current, H_b_current, v_b_current, t):
        M2 = mass_matrix_fun(H_b_current, q_current)[6:, 6:]
        h2 = bias_force_fun(H_b_current, q_current, v_b_current, dq_current)[6:]
        J_full = jacobian_fun(H_b_current, q_current)[0:3, 6:]
        end_effector_position = np.array(forward_kinematics_fun(H_b_current, q_current))[:3, 3]
        end_effector_velocity = J_full @ dq_current
        contact_force = compute_contact_force(end_effector_position, end_effector_velocity, k_friction, surface_height)
        f_ext = external_forces(t)
        try:
            result = np.linalg.solve(M2, tau_current - h2 + J_full.T @ f_ext[:3] + J_full.T @ contact_force[:3])
            return (
                np.clip(result.flatten(), -1e10, 1e10), 
                contact_force[:2], 
                end_effector_position, 
                end_effector_velocity, 
                contact_force
            )
        except Exception as e:
            return np.zeros_like(dq_current), np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3)

    def rk4_step(dq, q, tau, H_b, v_b, t, dt):
        # k1
        acc1, _, _, _, _ = acceleration(dq, q, tau, H_b, v_b, t)
        k1_dq = dt * acc1
        k1_q = dt * dq

        # k2
        dq_temp = dq + 0.5 * k1_dq
        q_temp = q + 0.5 * k1_q
        acc2, _, _, _, _ = acceleration(dq_temp, q_temp, tau, H_b, v_b, t + 0.5 * dt)
        k2_dq = dt * acc2
        k2_q = dt * dq_temp

        # k3
        dq_temp = dq + 0.5 * k2_dq
        q_temp = q + 0.5 * k2_q
        acc3, _, _, _, _ = acceleration(dq_temp, q_temp, tau, H_b, v_b, t + 0.5 * dt)
        k3_dq = dt * acc3
        k3_q = dt * dq_temp

        # k4
        dq_temp = dq + k3_dq
        q_temp = q + k3_q
        acc4, _, end_effector_position, end_effector_velocity, _ = acceleration(dq_temp, q_temp, tau, H_b, v_b, t + dt)
        k4_dq = dt * acc4
        k4_q = dt * dq_temp

        # Update dq and q
        dq_next = dq + (k1_dq + 2 * k2_dq + 2 * k3_dq + k4_dq) / 6
        q_next = q + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6

        return dq_next, q_next, end_effector_position, end_effector_velocity

    for i in range(N):
        t = i * dt
        
        q_des_current = get_q_des_current(i)
        controller.q_des = q_des_current

        # Control
        tau[:, i] = controller.control(q[:, i], dq[:, i])

        # Apply RK4 integration step
        dq[:, i+1], q[:, i+1], end_effector_position, end_effector_velocity = rk4_step(dq[:, i], q[:, i], tau[:, i], H_b, v_b, t, dt)

        # Check for large values and stop the simulation if detected
        if np.any(np.abs(q[:, i+1]) > 1e3) or np.any(np.abs(dq[:, i+1]) > 1e3):
            print(f"Large values detected at step {i}. Stopping simulation.")
            break

        if np.any(np.isnan(dq[:, i+1])) or np.any(np.isnan(q[:, i+1])) or np.any(np.abs(dq[:, i+1]) > 1e10) or np.any(np.abs(q[:, i+1]) > 1e10):
            print(f"Numerical instability detected at step {i}")
            break

        # Store end-effector data for analysis
        if 2 <= t:
            velocities.append(np.linalg.norm(dq[:, i+1]))
            positions.append(end_effector_position[2])
            end_effector_velocities.append(end_effector_velocity)
            ee_positions.append(end_effector_position)
            ee_velocities.append(end_effector_velocity)

    mean_velocity = end_effector_velocities[-1] if end_effector_velocities else 0
    return q, dq, k_friction, mean_velocity, end_effector_velocity, ee_positions, ee_velocities, friction_forces

def range_k_friction(velocity_threshold=0.01, max_k=1e4, num_steps=10, max_iterations=1000):
    k_friction_values = np.linspace(100, max_k, num_steps)
    k_max = None
    k_min = None
    all_friction_forces = []

    for iteration, k_friction in enumerate(k_friction_values):
        if iteration >= max_iterations:
            print(f"Reached maximum iterations ({max_iterations}). Stopping search.")
            break

        # Escludi il valore problematico di k_friction
        if k_friction == 12800:
            print(f"Skipping k_friction value: {k_friction} due to known instability.")
            continue

        try:
            _, _, _, mean_velocity, _, _, _, friction_forces = simulation(k_friction)
            all_friction_forces.append((k_friction, friction_forces))
        except ValueError as e:
            print(f"Numerical instability detected at k_friction: {k_friction}. Error: {e}")
            continue

        print(f"Iteration {iteration + 1}, k_friction: {k_friction}, Mean velocity: {mean_velocity}")

        if k_max is None and np.linalg.norm(mean_velocity) < velocity_threshold:
            k_max = k_friction
        elif k_min is None and np.linalg.norm(mean_velocity) > velocity_threshold:
            k_min = k_friction

        if k_max is not None and k_min is not None:
            break

    # Esegui un'iterazione aggiuntiva per k_min con k_friction = 0
    try:
        _, _, _, mean_velocity_zero, _, _, _, _ = simulation(0)
        _, _, _, mean_velocity_k_min, _, _, _, _ = simulation(k_min)

        if np.abs(np.linalg.norm(mean_velocity_zero) - np.linalg.norm(mean_velocity_k_min)) < 0.1:
            print(f"Mean velocity with k_friction = 0: {mean_velocity_zero}")
            print(f"Mean velocity with k_min: {mean_velocity_k_min}")
        else:
            print("The difference in mean velocities is greater than 0.1")
    except ValueError as e:
        print(f"Numerical instability detected during additional iteration. Error: {e}")

    return k_max, k_min, all_friction_forces

# Esempio di utilizzo della funzione range_k_friction
k_max, k_min, all_friction_forces = range_k_friction(velocity_threshold=0.01, max_k=1e4, num_steps=10, max_iterations=1000)
print(f"Found k_max: {k_max}, k_min: {k_min}")

# Plotting the friction forces
plt.figure(figsize=(12, 8))
for k_friction, friction_forces in all_friction_forces:
    friction_forces = np.array(friction_forces)
    plt.plot(friction_forces[:, 0], label=f'k_friction={k_friction}')

plt.xlabel('Time step')
plt.ylabel('Friction force')
plt.legend()
plt.title('Friction forces for different k_friction values')
plt.show()

# Calculate tracking error
tracking_error = np.linalg.norm(q_des - q[:, -1])
error = np.linalg.norm(q_des - q[:, -1], ord=np.inf)
print(f"Tracking error (L2 norm): {tracking_error:.4f}")
print(f"Maximum error (L-inf norm): {error:.4f}")

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

print("Simulation completed.")
