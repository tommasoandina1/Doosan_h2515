from adam.casadi.computations import KinDynComputations
from adam.geometry import utils
import numpy as np
import casadi as cs
#import icub_models
from math import sqrt

urdf_path =  "/Users/tommasoandina/Doosan_h2515-1/h2515.blue.urdf" 
# The joint list
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# Specify the root link
root_link = 'base'

kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF

#lol
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


#Valori randomici
q_des = np.array([0.1, 0.2 , 0.5, 0.2, 0.1, 0])
H_b = np.eye(4)

v_b = np.zeros(6)
s = (np.random.rand(len(joints_name_list)) - 0.5) * 5
s_dot = (np.random.rand(len(joints_name_list)) - 0.5) * 5





M = kinDyn.mass_matrix_fun()
M2 = cs.DM(M(H_b, s))
M2 = M2[6:, 6:]
print('Mass Matrix', M2)

h = kinDyn.bias_force_fun()
h2 = cs.DM(h(H_b, s, v_b, s_dot))


h2 = h2[6:]
print('bias force', h2)

q = np.zeros(num_dof)

kp = 0.1 
kd = 2*sqrt(kp)
dt = 1.0 / 16.0 * 1e-3
total_time = 2.0 * 1e-3



dq = np.zeros(num_dof)
ddq = np.zeros(num_dof)


N = int(total_time / dt)

ctrl = Controller(kp, kd, dt, q_des)
simu = Simulator(q, dt, dq, ddq)

q_des_np = cs.DM(q_des).full().flatten()
print(q_des_np)


for i in range(N):
    tau = ctrl.control(simu.q, simu.dq)
    simu.simulate_q(tau, h2)
    simu.simulate_dq(tau, h2)
    simu.simulate_ddq(M2, tau, h2)
   
    
q_des_np = cs.DM(q_des).full().flatten()
simu_q_np = cs.DM(simu.q).full().flatten()

# Calcola l'errore medio all'infinito tra i vettori NumPy
errore_medio_infinito = np.max(np.abs(q_des_np - simu_q_np))

print(q_des_np)
print(simu_q_np)

print("Errore medio all'infinito:", errore_medio_infinito)
