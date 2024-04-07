import adam
from adam.casadi import KinDynComputations
import icub_models
import numpy as np
import casadi as cs

model_path = "/home/parallels/Desktop/adam-main/LINUX/h2515.blue.urdf"
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





# robot dynamics 
tau = M @ cs.vertcat(v_b_dot, s_ddot) + C + G #lol


tau_fun = cs.Function('tau_f', [H, s, v_b, s_dot, v_b_dot, s_ddot], [tau])

# robot dynamics with the bias Force
tau1 = M @ cs.vertcat(v_b_dot, s_ddot) + h
tau_fun_biasforce = cs.Function('tau_f_biasforce', [H, s, v_b, s_dot, v_b_dot, s_ddot], [tau1])

