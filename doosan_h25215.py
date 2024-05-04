import adam
from adam.casadi import KinDynComputations
#import icub_models
import numpy as np
import casadi as cs
from math import sqrt


model_path = "h2515.blue.urdf" #"/home/tommaso/Doosan_h2515/h2515.blue.urdf"
joints_name_list = 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6' 
root_link = 'base'

kinDyn = KinDynComputations(model_path, joints_name_list, root_link)
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
gravity_term_fun =  kinDyn.gravity_term_fun()
bias_force_fun = kinDyn.bias_force_fun()

M = mass_matrix_fun(H, s)
C = coriolis_term_fun(H, s, v_b, s_dot)
G = gravity_term_fun(H, s)
h = bias_force_fun(H, s, v_b, s_dot) 


class ControllerPD:
    def __init__(self, dt, Kp=0.1):
        self.dt = dt  * 1e-3
        self.Kp = Kp
        self.Kd = 2 * sqrt(Kp)
        self.prev_ddq = 0 


    def update(self, q, dq, q_des, ddq):
        q_next = q + self.dt * dq
        dq_next = dq + self.dt * self.prev_ddq
        tau = self.Kp * (q_des - q_next) - self.Kd * dq_next  
        return q_next, dq_next, tau

dt = 1/16  * 1e-3
controller = ControllerPD(dt)



q = 0  # Posizione iniziale
dq = 0  # Velocità iniziale
q_des = 2  # Posizione desiderata
ddq = 0  # Accelerazione iniziale

for _ in range(33):  # Ciclo per 2 secondi (2 / dt)
    q, dq, tau = controller.update(q, dq, q_des, ddq)
    ddq = cs.inv(M) @ (tau - h)
    print(f"q: {q:.4f}, dq: {dq:.4f}, tau: {tau}, ddq: {ddq}")
    controller.prev_ddq = ddq  
   
tau_fun = cs.Function('tau_f', [H, s, v_b, s_dot, v_b_dot, s_ddot], [tau])


