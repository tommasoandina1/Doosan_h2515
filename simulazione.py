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
T_SIMULATION = 8  # Total simulation time in seconds
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


# Initial position (0 seconds)
q_initial = np.array([0, 0, 0, 0, 0, 0])

# Position at 2 seconds (surface contact)
q_des = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0])

# Position from 2 to 5 seconds (vary x and y, keep z constant)
q_des1 = np.array([0.2, -1.2, 1.0, -0.5, 0.3, 0.2])

# Final position (8 seconds, same as 2 seconds)
q_des2 = np.array([0.1, -1.2, 1.0, -0.5, 0.3, 0])


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
surface_height = 1.2  # Height of the surface

def compute_contact_force(position, velocity, k_friction):
    k_contact = 1000  # Contact stiffness
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
    min_position = float('inf')

    for i in range(N):
        t = i * dt

        if t < 2:
            q_des_current = q_initial + (q_des - q_initial) * (t / 2)
        elif t < 5:
            q_des_current = q_des1
        else:
            q_des_current = q_des1 + (q_des2 - q_des1) * ((t - 5) / 3)

        controller.q_des = q_des_current

        # Control
        tau[:, i] = controller.control(q[:, i], dq[:, i])

        # Dynamics
        M2 = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
        h2 = bias_force_fun(H_b, q[:, i], v_b, dq[:, i])[6:]
        J_full = jacobian_fun(H_b, q[:, i])[0:3, 6:]

        # External forces
        end_effector_position = np.array(forward_kinematics_fun(H_b, q[:, i]))[:3, 3]
        end_effector_velocity = J_full @ dq[:, i]

        if 2 <= t <= 8:
            end_effector_position[2] = surface_height

        contact_force = compute_contact_force(end_effector_position, end_effector_velocity, k_friction)
        contact_force_dm = cs.DM(contact_force.flatten())
        f_ext = external_forces(t, end_effector_position, end_effector_velocity, k_friction)
        f_ext_dm = cs.DM(f_ext.flatten())

        if np.any(np.abs(q[:, i]) > 1e10) or np.any(np.abs(dq[:, i]) > 1e10):
            print(f"Extremely large values detected at step {i}. Stopping simulation.")
            break

        # RK4 integration
        def acceleration(dq_current):
            try:
                result = np.array(cs.solve(M2, tau[:, i] - h2 + cs.mtimes(J_full.T, f_ext_dm[:3]) + cs.mtimes(J_full.T, contact_force_dm[:3]) - cs.mtimes(M2, cs.DM(dq_current)))).reshape(-1)
                return np.clip(result, -1e10, 1e10)
            except Exception as e:
                print(f"Error in acceleration calculation at step {i}: {e}")
                return np.zeros_like(dq_current)

        k1 = dt * acceleration(dq[:, i])
        k2 = dt * acceleration(dq[:, i] + 0.5 * k1)
        k3 = dt * acceleration(dq[:, i] + 0.5 * k2)
        k4 = dt * acceleration(dq[:, i] + k3)

        # Update dq and q
        dq[:, i+1] = np.clip(dq[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6, -1e10, 1e10)
        q[:, i+1] = np.clip(q[:, i] + dt * dq[:, i], -1e10, 1e10)

        if np.any(np.isnan(dq[:, i+1])) or np.any(np.isnan(q[:, i+1])):
            print(f"NaN detected in dq or q at step {i}")
            break

        if 2 <= t <= 8:
            velocities.append(np.linalg.norm(dq[:, i+1]))
            positions.append(end_effector_position[2])

    # Calcola la velocità media e la posizione media
    mean_velocity = np.mean(velocities) if velocities else 0
    return q, dq, k_friction, mean_velocity, end_effector_velocity





def range_k_friction2(velocity_threshold=0.01, max_k=1e4, num_steps=10, max_iterations=1000):
    k_friction_values = np.linspace(100, max_k, num_steps)
    k_max = None
    k_min = 3

    for iteration, k_friction in enumerate(k_friction_values):
        if iteration >= max_iterations:
            print(f"Reached maximum iterations ({max_iterations}). Stopping search.")
            break

        _, _, _, mean_velocity, _ = simulation(k_friction)
        print(f"Iteration {iteration + 1}, k_friction: {k_friction}, Mean velocity: {mean_velocity}")
        if k_max is None and mean_velocity < velocity_threshold:
            k_max = k_friction
            print(f"Found k_max: {k_friction}, Mean velocity: {mean_velocity}")

        if k_max is not None and k_min is not None:
            break

    return k_max, k_min


print("Starting search for k_max and k_min...")

k_max, k_min = range_k_friction2()
print(f"Final k_max: {k_max}, k_min: {k_min}")
'''
def range_k_friction2(velocity_threshold=0.01, max_k=1e4, num_steps=10):
    k_friction_values = np.linspace(100, max_k, num_steps)
    #_, _, _, frictionless_velocity, _ = simulation(0)
    k_max = None
    k_min = 3 #None

    for k_friction in k_friction_values:
        _, _, _, mean_velocity, _ = simulation(k_friction)
        if k_max is None and mean_velocity < velocity_threshold:
            k_max = k_friction
            print(f"Found k_max: {k_friction}, Mean velocity: {mean_velocity}")
        

        if k_min is None and frictionless_velocity - mean_velocity < velocity_threshold:
            k_min = k_friction
            print(f"Found k_min: {k_friction}, Mean velocity: {mean_velocity}")


        # If both k_max and k_min are found, no need to continue
        if k_max is not None and k_min is not None:
            break
        #k_min = 3
    
    return k_max, k_min

# Call the function and print results
k_max, k_min = range_k_friction2()
print(f"Final k_max: {k_max}, k_min: {k_min}")

'''


'''

# Define k_friction_values
if k_min is not None and k_max is not None:
    k_friction_values = np.logspace(np.log10(k_min), np.log10(k_max), 100)
else:
    k_friction_values = np.logspace(-1, 3, 100)  # Default range if k_min or k_max is None

# Plot k_friction range vs mean velocity
plt.figure(figsize=(12, 8))
plt.plot(k_friction_values, [simulation(k)[3] for k in k_friction_values], label='Mean Velocity')
if k_max is not None:
    plt.axvline(k_max, color='r', linestyle='--', label=f'k_max = {k_max:.2f}')
if k_min is not None:
    plt.axvline(k_min, color='g', linestyle='--', label=f'k_min = {k_min:.2f}')
plt.xlabel('k_friction')
plt.ylabel('Mean Velocity')
plt.title('k_friction Range vs Mean Velocity')
plt.legend()
plt.grid(True)
plt.xscale('log')
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

print(f"Optimal kp: {optimal_kp} with tracking error {min_tracking_error}")
print(f"Optimal kd: {optimal_kd} with infinity norm error {error}")
print("Simulation completed.")
'''
