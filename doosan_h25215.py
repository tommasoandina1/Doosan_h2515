import adam
from adam.casadi import KinDynComputations
#import icub_models
import numpy as np
import casadi as cs
from math import sqrt


model_path = "h2515.blue.urdf" 
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
Jacobian_fun = kinDyn.jacobian_fun("link6")


M = mass_matrix_fun(H, s)
C = coriolis_term_fun(H, s, v_b, s_dot)
G = gravity_term_fun(H, s)
h = bias_force_fun(H, s, v_b, s_dot)
J = Jacobian_fun(H, s)
new_J = J[:3, :] #first three rows

#Dissiaptive Force

a = 40
f_lev = cs.SX.sym('f_lev')
f_min = 10
f_max = 200

f = cs.linspace(f_min, f_max, 100)  # Genera 100 valori nel range [10, 200]

# Calcola le forze f_vec in base alla frequenza f
f_vec = cs.vertcat(a*120 * cs.sin(2 * cs.pi * f),
                   a*70 * cs.sin(2 * cs.pi * f),
                   a*40 * cs.sin(2 * cs.pi * f))

# Creare una funzione per valutare le forze in base alla frequenza
f_vec_fun = cs.Function('f_vec_fun', [f_lev], [f_vec])

#Controll law (tau)
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

        q = q_next
        dq = dq_next
        return q, dq, tau



dt_initial = 1/16 * 1e-3  
dt_final = 2 * 1e-3  
num_steps = int(dt_final/dt_initial)  # Numero di passi


dt = dt_initial
errors = []

controller = ControllerPD(dt_initial)



# Inizializza q alla posizione iniziale del robot
q = 0  
dq = 0
q_des = 2
ddq = 0

while dt <= dt_final:
    errors_per_iteration = []

    for _ in range(num_steps):
        q, dq, tau = controller.update(q, dq, q_des, ddq)
        ddq = cs.inv(M) @ (tau - h)

        # Stampa il valore di tau ad ogni iterazione
        print(f"tau: {tau}")

    # Calcola l'errore finale e lo salva
    final_error = abs(q_des - q)
    errors.append(final_error)

    dt *= 2

# Stampare tutti i valori dell'errore
for iteration, error in enumerate(errors, 1):
    print(f"Iterazione {iteration}: Errore finale: {error}")


"""
dt = dt_initial
errors = []

while dt <= dt_final:
    q = 0
    dq = 0
    q_des = 20
    ddq = 0
    mean_error = 0

    for _ in range(num_steps):
        q, dq, tau = controller.update(q, dq, q_des, ddq)
        
        
        f_vec_val = f_vec_fun(f_lev)  
        
        
        f_vec_val_3x1 = cs.vertcat(f_vec_val[0], f_vec_val[1], f_vec_val[2])
        
        
        ddq = cs.inv(M) @ (tau - h + new_J.T @ f_vec_val_3x1)  

    errors.append(mean_error)
    dt *= 2

mean_error_inf_norm = max(errors)
print(f"Mean error inf-norm con forze: {mean_error_inf_norm}")

"""