import numpy as np
from numpy import nan
import casadi as cs
from numpy.linalg import norm
from scipy.interpolate import CubicSpline

# Parameters for contact
k_contatto = 3e4  # Rigidezza del contatto
d_contatto = np.sqrt(k_contatto)  # Coefficiente di smorzamento del contatto
z_superficie = 1  # Altezza della superficie
amplitudes = np.array([50, 60, 90])  # Accelerations amplitudes in x, y, z
frequency_range = (10, 200)  # Frequencies range in Hz



class Simulator:
    def __init__(self, num_dof, mass_matrix_fun, bias_force_fun, jacobian_fun, forward_kinematics_fun, t, metodo_integrazione='RK4'):
        self.num_dof = num_dof
        self.mass_matrix_fun = mass_matrix_fun
        self.bias_force_fun = bias_force_fun
        self.jacobian_fun = jacobian_fun
        self.forward_kinematics_fun = forward_kinematics_fun
        self.q = np.zeros(num_dof)
        self.v = np.zeros(num_dof)
        self.tau_c = np.zeros(num_dof)
        self.dv = np.zeros(num_dof)
        self.t = t
        self.k_attrito = None
        self.metodo_integrazione = metodo_integrazione
    
    def set_k_attrito(self, k_attrito):
        self.k_attrito = k_attrito

    def init(self, q0):
        self.q = q0
        self.v = np.zeros_like(q0)

    def dynamics(self, q, v, tau):
        H_b = np.eye(4)
        v_b = np.zeros(6)

        M = self.mass_matrix_fun(H_b, q)[6:12, 6:12]
        h_full = self.bias_force_fun(H_b, q, v_b, v)
        h = h_full[6:12]

        J_full = self.jacobian_fun(H_b, q)[0:3, 6:12]
        end_effector_position = np.array(self.forward_kinematics_fun(H_b, q))[:3, 3]
        end_effector_velocity = np.array((J_full @ v).full()).flatten()

        # Calcolo delle forze esterne
        z_ee = end_effector_position[2]
        vz_ee = end_effector_velocity[2]
        F_normale = calcola_forza_normale(z_ee, vz_ee)
        F_attrito = calcola_forza_attrito(F_normale, end_effector_velocity[:2].flatten(), self.k_attrito)
        F_grinder = external_forces(self.t)  
        F_tot = F_normale + F_attrito + F_grinder

        ddq = cs.solve(M, tau - h + J_full.T @ F_tot)
        return ddq

    def euler_step(self, tau, dt):
        self.dv = self.dynamics(self.q, self.v, tau)
        self.v += self.dv * dt
        self.q += self.v * dt

    def rk4_step(self, tau, dt):
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

    def simulate(self, tau, dt, ndt):
        for _ in range(ndt):
            if self.metodo_integrazione == 'Euler':
                self.euler_step(tau, dt)
            else:  # Default to RK4
                self.rk4_step(tau, dt)

            if np.any(np.isnan(self.q)) or np.any(np.isnan(self.v)):
                raise ValueError("NaN detected in simulator state after integration")

            self.q = np.clip(self.q, -1e3, 1e3)
            self.v = np.clip(self.v, -1e3, 1e3)

    def get_ddq(self, tau):
        ddq = self.dynamics(self.q, self.v, tau)
        return np.array(ddq).flatten()


def calcola_forza_normale(z_ee, vz_ee):
    F_normale = np.zeros(3)
    if z_ee < z_superficie:
        F_normale[2] = k_contatto * (z_superficie - z_ee) - d_contatto * vz_ee
    return F_normale

def calcola_forza_attrito(F_normale, v_tangenziale, k_attrito):
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


def generate_sin_trajectory(t, x0, amp, phi, freq):

    two_pi_f = 2 * np.pi * freq                  
    two_pi_f_amp = two_pi_f * amp
    two_pi_f_squared_amp = two_pi_f * two_pi_f_amp

    
    x_ref = x0 + amp * np.sin(two_pi_f * t + phi)
    dx_ref = two_pi_f_amp * np.cos(two_pi_f * t + phi)
    ddx_ref = -two_pi_f_squared_amp * np.sin(two_pi_f * t + phi)
    
    return x_ref, dx_ref, ddx_ref


def generate_trajectory(t):
    t_points = [0.0, 6.0]
    x_points = [
        [1.50882228e-04, 1.02054791e-02, 1.84429995e+00],  
        [1, 1, 0.8],  

    ]

    x_spline = CubicSpline(t_points, [p[0] for p in x_points], bc_type='natural')
    y_spline = CubicSpline(t_points, [p[1] for p in x_points], bc_type='natural')
    z_spline = CubicSpline(t_points, [p[2] for p in x_points], bc_type='natural')

    x_ref = np.array([x_spline(t), y_spline(t), z_spline(t)])
    dx_ref = np.array([x_spline(t, 1), y_spline(t, 1), z_spline(t, 1)])
    ddx_ref = np.array([x_spline(t, 2), y_spline(t, 2), z_spline(t, 2)])

    return x_ref, dx_ref, ddx_ref
