from adam.casadi import KinDynComputations
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
M = M[:6, :6]
C = coriolis_term_fun(H, s, v_b, s_dot)
G = gravity_term_fun(H, s)
h = bias_force_fun(H, s, v_b, s_dot)
h = h[:6]
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

    def simulate_q(self, tau, h):
        dq = self.simulate_dq(tau, h)
        self.q += self.dt * dq
        return self.q
    
    def simulate_dq(self, tau, h):
        self.ddq = cs.inv(M) @ (tau - h)
        self.dq += self.dt * self.ddq
        return self.dq
    
    def simulate_ddq(self, M, tau, h):
        self.ddq = cs.inv(M) @ (tau - h)
        return self.ddq

q_0 = cs.SX.sym('q_0', num_dof)
kp = 0.1 
kd = sqrt(kp)
dt = 1.0 / 16.0 
total_time = 2.0 
q_des = cs.SX.sym('q_des', num_dof)
dq = cs.SX.sym('dq', num_dof)
ddq = cs.SX.sym('ddq', num_dof)

N = int(total_time / dt)

ctrl = Controller(kp, kd, dt, q_des)
simu = Simulator(q_0, dt, dq, ddq)

for i in range(N):
    print("Time", i * dt, "q =", simu.q)
    
    M = mass_matrix_fun(H, s)  # Assicurati che M sia definito prima di essere utilizzato
    M = M[:6, :6]
    h = bias_force_fun(H, s, v_b, s_dot)  # Calcola il termine h per la configurazione attuale del sistema
    h = h[:6]

    tau = ctrl.control(simu.q, simu.dq)
    simu.simulate_q(tau, h)
    simu.simulate_ddq(M, tau, h)

