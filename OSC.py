import numpy as np
from numpy import nan
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time
from scipy.interpolate import CubicSpline


# Configuration
urdf_path = "/Users/tommasoandina/Desktop/Doosan_h2515-main/model.urdf"
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
root_link = 'base'
end_effector = 'link6'

# Parameters for contact
k_contatto = 3e4  # Rigidezza del contatto
d_contatto = np.sqrt(k_contatto)    # Coefficiente di smorzamento del contatto
z_superficie = 1.0   # Altezza della superficie
amplitudes = np.array([50, 60, 90])  # Accelerations amplitudes in x, y, z
frequency_range = (10, 200)  # Frequencies range in Hz



# Initialize KinDynComputations
kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF

# Dynamics functions
mass_matrix_fun = kinDyn.mass_matrix_fun()
bias_force_fun = kinDyn.bias_force_fun()
jacobian_fun = kinDyn.jacobian_fun(end_effector)
forward_kinematics_fun = kinDyn.forward_kinematics_fun(end_effector)


tests = []
tests += [{'controllore': 'OSC',  'k_attrito': 0}]
tests += [{'controllore': 'OSC',  'k_attrito': 10}]
tests += [{'controllore': 'OSC', 'k_attrito': 50}]
tests += [{'controllore': 'OSC', 'k_attrito': 100}]



# Simulation parameters
conf = {
    'T_SIMULATION': 6.0,
    'dt_control': 1/1000,   
    'dt_simulation': 1/16000,  
    'PRINT_T': 1.0,
    'simulate_real_time': False,
    'kp_j': 100.0,
    'kd_j': np.sqrt(100),
    'kp': 1000,  
    'kd': 2 * np.sqrt(1000), 
    'q0': np.zeros(num_dof),
}

# Placeholder for simulator
class Simulator:
    def __init__(self):
        self.q = np.zeros(num_dof)
        self.v = np.zeros(num_dof)
        self.tau_c = np.zeros(num_dof)
        self.dv = np.zeros(num_dof)

    def init(self, q0):
        self.q = q0
        self.v = np.zeros_like(q0)

    
    def dynamics(self, q, v, tau):
        H_b = np.eye(4)
        v_b = np.zeros(6)

        M = mass_matrix_fun(H_b, q)[6:12, 6:12]
        h_full = bias_force_fun(H_b, q, v_b, v)
        h = h_full[6:12]
    
        J_full = jacobian_fun(H_b, q)[0:3, 6:12]
        end_effector_position = np.array(forward_kinematics_fun(H_b, q))[:3, 3]
        end_effector_velocity = np.array((J_full @ v).full()).flatten()  

            
        # Calcolo delle forze esterne
        z_ee = end_effector_position[2]
        vz_ee = end_effector_velocity[2]
        F_normale = calcola_forza_normale(z_ee, vz_ee)
        F_attrito = calcola_forza_attrito(F_normale, end_effector_velocity[:2].flatten())  
        F_grinder = external_forces(t)
        F_tot = F_normale + F_attrito + F_grinder

        ddq = cs.solve(M, tau - h + J_full.T @ F_tot)
        return ddq

    def simulate(self, tau, dt, ndt):
        for _ in range(ndt):
            k1_v = self.dynamics(self.q, self.v, tau) * dt
            k1_q = self.v * dt

            q2 = self.q + 0.5 * k1_q
            v2 = self.v + 0.5 * k1_v
            k2_v = self.dynamics(q2, v2, tau) * dt
            k2_q = v2 * dt

            q3 = self.q + 0.5 * k2_q
            v3 = self.v + 0.5 * k2_v
            k3_v = self.dynamics(q3, v3, tau) * dt
            k3_q = v3 * dt

            q4 = self.q + k3_q
            v4 = self.v + k3_v
            k4_v = self.dynamics(q4, v4, tau) * dt
            k4_q = v4 * dt

            self.v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
            self.q += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6

            if np.any(np.isnan(self.q)) or np.any(np.isnan(self.v)):
                raise ValueError("NaN detected in simulator state after integration")

            self.q = np.clip(self.q, -1e3, 1e3)
            self.v = np.clip(self.v, -1e3, 1e3)
    
    def get_ddq(self, tau):
        return self.dynamics(self.q, self.v, tau)



simu = Simulator()

def generate_trajectory(t):
    t_points = [0.0, 6.0]
    x_points = [
        [1.50882228e-04, 1.02054791e-02, 1.84429995e+00],  
        [1, 1, 0.8],  

    ]

    # Ensure continuity by using cubic spline interpolation
    x_spline = CubicSpline(t_points, [p[0] for p in x_points], bc_type='natural')
    y_spline = CubicSpline(t_points, [p[1] for p in x_points], bc_type='natural')
    z_spline = CubicSpline(t_points, [p[2] for p in x_points], bc_type='natural')

    x_ref = np.array([x_spline(t), y_spline(t), z_spline(t)])
    dx_ref = np.array([x_spline(t, 1), y_spline(t, 1), z_spline(t, 1)])
    ddx_ref = np.array([x_spline(t, 2), y_spline(t, 2), z_spline(t, 2)])

    return x_ref, dx_ref, ddx_ref


def calcola_forza_normale(z_ee, vz_ee):
    F_normale = np.zeros(3)
    if z_ee < z_superficie:  
        F_normale[2] = k_contatto * (z_superficie - z_ee) - d_contatto * vz_ee
    return F_normale

def calcola_forza_attrito(F_normale, v_tangenziale):
    F_attrito = np.zeros(3)
    if np.linalg.norm(v_tangenziale) > 0:
        F_attrito[:2] = -k_attrito * F_normale[2] * (v_tangenziale / np.linalg.norm(v_tangenziale))
    return F_attrito


def external_forces(t):

    forces = np.zeros(3)
    frequencies = np.linspace(frequency_range[0], frequency_range[1], len(amplitudes))

    for i, (amp, freq) in enumerate(zip(amplitudes, frequencies)):
        forces[i] = amp * np.sin(2 * np.pi * freq * t)

    return forces



# Simulazione principale con time step diversi
kp = conf['kp']
kd = conf['kd']

tracking_err_osc = []




for (test_id, test) in  enumerate(tests):
    description = str(test_id)+' Controller '+test['controllore']+' k_frizione='+str(test['k_frizione'])
    print(description)

    k_attrito = test['k_attrito'']
    simu.init(conf['q0'])
    ddq_totali = []
    
    nx, ndx = 3, 3
    N_control = int(conf['T_SIMULATION'] / conf['dt_control'])
    N_simulation = int(conf['dt_control'] / conf['dt_simulation'])
    tau = np.empty((num_dof, N_control)) * nan
    tau_c = np.empty((num_dof, N_control)) * nan
    q = np.empty((num_dof, N_control + 1)) * nan
    v = np.empty((num_dof, N_control + 1)) * nan
    dv = np.empty((num_dof, N_control + 1)) * nan
    x = np.empty((nx, N_control)) * nan    
    dx = np.empty((ndx, N_control)) * nan
    ddx = np.empty((ndx, N_control)) * nan
    x_ref = np.empty((nx, N_control)) * nan
    dx_ref = np.empty((ndx, N_control)) * nan
    ddx_ref = np.empty((ndx, N_control)) * nan
    ddx_des = np.empty((ndx, N_control)) * nan

    t = 0.0
    PRINT_N = int(conf['PRINT_T'] / conf['dt_control'])



    for i in range(0, N_control):
        time_start = time.time()



        x_ref[:, i], dx_ref[:, i], ddx_ref[:, i] = generate_trajectory(t)

        v[:, i] = np.squeeze(simu.v)
        q[:, i] = np.squeeze(simu.q)

            
        try:
            H_b = np.eye(4)
            v_b = np.zeros(6)
            dq = v[:, i]

            M = mass_matrix_fun(H_b, q[:, i])[6:, 6:]
            h = bias_force_fun(H_b, q[:, i], v_b, dq)[6:12]
            J_full = jacobian_fun(H_b, q[:, i])[0:3, 6:]

            end_effector_position = np.array(forward_kinematics_fun(H_b, q[:, i]))[:3, 3]
            end_effector_velocity = np.array(J_full @ dq).flatten()

            
            dJdq = np.zeros(3)
            delta = 1e-6
            for j in range(num_dof):
                q_plus = q[:, i].copy()
                q_minus = q[:, i].copy()
                q_plus[j] += delta
                q_minus[j] -= delta
                J_plus = jacobian_fun(H_b, q_plus)[0:3, 6:]
                J_minus = jacobian_fun(H_b, q_minus)[0:3, 6:]
                dJdq += (J_plus - J_minus) @ dq / (2 * delta)

            x[:, i] = end_effector_position
            dx[:, i] = np.array(end_effector_velocity).flatten()

            ddx_fb = kp * (x_ref[:, i] - x[:, i]) + kd * (dx_ref[:, i] - dx[:, i])
            ddx_des[:, i] = ddx_ref[:, i] + ddx_fb

            Minv = cs.inv(M)
            J_Minv = J_full @ Minv
            Lambda = cs.inv(J_Minv @ J_full.T)

            mu = Lambda @ (J_Minv @ h - dJdq)
            f = Lambda @ ddx_des[:, i] + mu
            

            J_T_pinv = Lambda @ J_Minv
            NJ = np.eye(num_dof) - J_full.T @ J_T_pinv
            tau_0 = M @ (conf['kp_j'] * (conf['q0'] - q[:, i]) - conf['kd_j'] * v[:, i])
            tau[:, i] = np.squeeze(J_full.T @ f) + np.squeeze(NJ @ (tau_0 + h))
        


            # Simulazione con il passo del simulatore
            simu.simulate(tau[:, i], conf['dt_simulation'], N_simulation)
            tau_c[:, i] = simu.tau_c
            dv[:, i] = simu.dv
            t += conf['dt_control']

            if end_effector_position[2] < z_superficie:
                ddq = simu.get_ddq(tau[:, i])
                ddq_totali.append(ddq)

     
            time_spent = time.time() - time_start
            if conf['simulate_real_time'] and time_spent < conf['dt_control']:
                time.sleep(conf['dt_control'] - time_spent)

        except Exception as e:
            print(f"Error at step {i}: {e}")
            print(f"q = {q[:, i]}")
            print(f"v = {v[:, i]}")
            print(f"x = {x[:, i]}")
            print(f"dx = {dx[:, i]}")
            break

    tracking_err = np.max(np.linalg.norm(x_ref - x, axis=0, ord=np.inf))
    tracking_err_osc.append({'value': tracking_err})    
    print('Max tracking error (norma all\'infinito): %.3f m\n' % (tracking_err))

    ddq_totali = np.array(ddq_totali)
    accelerazione_media = np.mean(np.abs(ddq_totali))
    print('Accelerazione media: %.3f m/s^2\n' % (accelerazione_media))


    # Plot delle Traiettorie dei Giunti
    (f, ax) = plt.subplots(num_dof, 1, figsize=(10, 20))
    tt = np.arange(0.0, N_control * conf['dt_control'], conf['dt_control'])

    for i in range(num_dof):
        ax[i].plot(tt, q[i, :-1], label=f'q_{i}')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(f'q_{i} [rad]')
        ax[i].legend()
    plt.suptitle(f'Traiettorie dei Giunti per k_friction = {test['k_frizione']}')
    plt.show()

    # Plot delle Traiettorie Cartesiane
    plt.figure()
    for i in range(nx):
        plt.plot(tt, x[i, :], label=f'x_{i}')
        plt.plot(tt, x_ref[i, :], '--', label=f'x_ref_{i}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title(f'Traiettorie Cartesiane per k_frition = {test['k_frizione']}')
    plt.legend()
    plt.show()

