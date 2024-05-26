from adam.casadi.computations import KinDynComputations
from adam.geometry import utils
import numpy as np
import casadi as cs
from math import sqrt
import matplotlib.pyplot as plt

urdf_path = "/Users/tommasoandina/Doosan_h2515/h2515.blue.urdf"
# The joint list
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# Specify the root link
root_link = 'base'

kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF




H = cs.SX.sym('H', 4, 4)
# The joint values
s = cs.SX.sym('s', num_dof)
# The base velocity
v_b = cs.SX.sym('v_b', 6)
# The joints velocity
s_dot = cs.SX.sym('s_dot', num_dof)
# The base acceleration
v_b_dot = cs.SX.sym('v_b_dot', 6)
# The joints acceleration
s_ddot = cs.SX.sym('s_ddot', num_dof)

# initialize
mass_matrix_fun = kinDyn.mass_matrix_fun()
coriolis_term_fun = kinDyn.coriolis_term_fun()
gravity_term_fun = kinDyn.gravity_term_fun()
bias_force_fun = kinDyn.bias_force_fun()
Jacobian_fun = kinDyn.jacobian_fun("link6")

class Controller:
    def __init__(self, kp, kd, dt, q_des):
        self.q_previous = 0.0
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.q_des = q_des
        self.first_iter = True

    def control(self, q, dq):
        if self.first_iter:
            self.q_previous = q
            self.first_iter = False

        self.q_previous = q
        return self.kp * (self.q_des - q) - self.kd * dq

class Simulator:
    def __init__(self, q, dt, dq, ddq):
        self.q = q
        self.dt = dt
        self.dq = dq
        self.ddq = ddq

    def simulate_q(self, tau, h2):
        dq = self.simulate_dq(tau, h2)
        self.q += self.dt * dq
        return self.q

    def simulate_dq(self, tau, h2):
        self.ddq = cs.inv(M2) @ (tau - h2)
        self.dq += self.dt * self.ddq
        return self.dq

    def simulate_ddq(self, M2, tau, h2):
        self.ddq = cs.inv(M2) @ (tau - h2)
        return self.ddq

# Valori randomici
q_des = np.array([0.1, 0.2 , 0.25, 0.2, 0.1, 0.0])
H_b = np.eye(4)

v_b = np.zeros(6)
s = (np.random.rand(len(joints_name_list)) - 0.5) * 5
s_dot = (np.random.rand(len(joints_name_list)) - 0.5) * 5

M = kinDyn.mass_matrix_fun()
M2 = cs.DM(M(H_b, s))
M2 = M2[6:, 6:]

h = kinDyn.bias_force_fun()
h2 = cs.DM(h(H_b, s, v_b, s_dot))
h2 = h2[6:]

q = np.zeros(num_dof)

initial_kp =   500
kp_increment = 1
max_kp = 1000
kd = 2 * sqrt(initial_kp)
dt = 1.0 / 16.0 * 1e-3
total_time = 2 * 1e-3

dq = np.zeros(num_dof)
ddq = np.zeros(num_dof)

N = int(total_time / dt)



# plot dati
time_values = []
q_values = [[] for _ in range(num_dof)]

ctrl = Controller(initial_kp, kd, dt, q_des)
simu = Simulator(q, dt, dq, ddq)

q_des_np = cs.DM(q_des).full().flatten()

for i in range(N):
    tau = ctrl.control(simu.q, simu.dq)
    simu.simulate_q(tau, h2)
    simu.simulate_dq(tau, h2)
    simu.simulate_ddq(M2, tau, h2)

    simu_q_np = np.array(simu.q).flatten()
    q_des_np = np.array(q_des).flatten()

    # Registrare i dati per il plot
    time_values.append(i * dt)
    for j in range(num_dof):
        q_values[j].append(simu.q[j].full().item())  

    # Aumentare kp
    if ctrl.kp < max_kp:
        ctrl.kp += kp_increment
        ctrl.kd = 2 * sqrt(ctrl.kp)

    if np.allclose(simu_q_np, q_des_np, atol=1e-3):
        break


# norme
#def calculate_norm(vector1, vector2):
#    return np.linalg.norm(vector1 - vector2)

#norm = calculate_norm(q_des_np, simu_q_np)
#print("Norma tra q_des e q:", norm)

# Plotting
fig, axs = plt.subplots(num_dof, 1, figsize=(10, 15))
for j in range(num_dof):
    kp_values = [initial_kp + i * kp_increment for i in range(len(q_values[j]))]
    axs[j].plot(kp_values, q_values[j], label=f'q{j+1}')
    axs[j].axhline(y=q_des[j], color='r', linestyle='--', label=f'q{j+1}_des')
    axs[j].set_xlabel('kp')
    axs[j].set_ylabel(f'q{j+1}')
    axs[j].legend()
    axs[j].grid()

plt.tight_layout()
plt.show()
