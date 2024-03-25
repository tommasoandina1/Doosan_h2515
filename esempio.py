import adam
from adam.casadi import KinDynComputations
from adam.geometry import utils
import numpy as np
import casadi as cs

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = "/home/parallels/Desktop/adam-main/LINUX/h2515.blue.urdf"
# The joint list
joints_name_list = [
   'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6' 
]
# Specify the root link
root_link = 'base'

kinDyn = KinDynComputations(model_path, joints_name_list, root_link)


# Set joints and base informations
xyz = (np.random.rand(3) - 0.5) * 5
rpy = (np.random.rand(3) - 0.5) * 5
H_b = utils.H_from_Pos_RPY(xyz, rpy)
v_b = (np.random.rand(6) - 0.5) * 5
s = (np.random.rand(len(joints_name_list)) - 0.5) * 5
s_dot = (np.random.rand(len(joints_name_list)) - 0.5) * 5

# Matrice di Massa
M = kinDyn.mass_matrix_fun()
print('Mass matrix:\n', cs.DM(M(H_b, s)))

# Bias Force
h = kinDyn.bias_force_fun() 
print('Bias Force:\n', cs.DM(h(H_b, s, v_b, s_dot)))

# Coriolis Terms
C = kinDyn.coriolis_term_fun()
print('Coriolis Terms:\n', cs.DM(C(H_b, s, v_b, s_dot)))

# Acceleration Terms
ddq = kinDyn.ddq_fun()
print('ddq term:\n', cs.DM(ddq(H_b, s, v_b, s_dot, M, h, C)))
