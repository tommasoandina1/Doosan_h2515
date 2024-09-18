import numpy as np
from numpy import nan
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time, sys


from function import Simulator, z_superficie, generate_sin_trajectory

# Configuration
urdf_path = "/Users/tommasoandina/Desktop/DOOS/model.urdf"
joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
root_link = 'base'
end_effector = 'link6'

# Initialize KinDynComputations
kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
num_dof = kinDyn.NDoF

# Dynamics functions
mass_matrix_fun = kinDyn.mass_matrix_fun()
bias_force_fun = kinDyn.bias_force_fun()
jacobian_fun = kinDyn.jacobian_fun(end_effector)
forward_kinematics_fun = kinDyn.forward_kinematics_fun(end_effector)




tests = []

acceleration_results = {}
#k_attrito_values = [0.0, 0.05] #, 0.10, 0.15, 0.20]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 1.25, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.05, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.1, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.2, 'metodo_integrazione': 'RK4'}]


#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.1, 'metodo_integrazione': 'Euler'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

#tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
#tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]






# Simulation parameters
conf = {
    'T_SIMULATION': 6.0,
    'dt_control': 1/1000,   
    'PRINT_T': 1.0,
    'simulate_real_time': False,
    'kp_j': 100.0,
    'kd_j': np.sqrt(100),
    'kp': 1000,  
    'kd':  np.sqrt(1000), 
    'q0': np.zeros(num_dof),
}





tracking_err_osc_RK4 = []
tracking_err_ic_RK4  = []
tracking_err_osc_Euler = []
tracking_err_ic_Euler = []
inf_norm_errors_osc_RK4 = []
inf_norm_errors_ic_RK4 = []
inf_norm_errors_osc_Euler = []
inf_norm_errors_ic_Euler = []

dt_simulation_values = []



N_control = int(conf['T_SIMULATION'] / conf['dt_control'])

errors = []

for (test_id, test) in enumerate(tests):
    print()
    description = f"{test['controllore']} kp={test['kp']} dt_simulation={test['dt_simulation']} k_attrito={test['k_attrito']} metodo={test['metodo_integrazione']}"
    print(description)
    
    simu = Simulator(num_dof, mass_matrix_fun, bias_force_fun, jacobian_fun, forward_kinematics_fun, conf['T_SIMULATION'], test['metodo_integrazione'])
    kp = test['kp']
    kd = 2 * np.sqrt(kp)
    k_attrito = test['k_attrito']
    simu.set_k_attrito(k_attrito)
    simu.init(conf['q0'])
    
    metodo_integrazione = test['metodo_integrazione']
    
    accelerazioni = []
    velocita = []
    instabile = False
    bloccato = False
    
    nx, ndx = 3, 3
    N_simulation = int(conf['dt_control'] / test['dt_simulation'])
    
    tau = np.full((num_dof, N_control), nan)
    tau_c = np.full((num_dof, N_control), nan)
    q = np.full((num_dof, N_control + 1), nan)
    v = np.full((num_dof, N_control + 1), nan)
    dv = np.full((num_dof, N_control + 1), nan)
    x = np.full((nx, N_control), nan)
    dx = np.full((ndx, N_control), nan)
    ddx = np.full((ndx, N_control), nan)
    x_ref = np.full((nx, N_control), nan)
    dx_ref = np.full((ndx, N_control), nan)
    ddx_ref = np.full((ndx, N_control), nan)
    ddx_des = np.full((ndx, N_control), nan)
    accelerazione_ee_xy = np.full((2, N_control), nan)  # Solo per x e y
    
    x0 = np.array([0.032, 0.09, 1]).T  # offset
    amp = np.array([0.3, 0.5, 0.15]).T
    phi = np.array([0.0, 0, 0]).T  # phase
    freq = np.array([0.5, 0.5, 0.5]).T
    
    t = 0.0
    PRINT_N = int(conf['PRINT_T'] / conf['dt_control'])
    
    for i in range(0, N_control):
        time_start = time.time()
        ddx_actual = np.zeros(3)
        
        # Imposta la traiettoria di riferimento
        x_ref[:, i], dx_ref[:, i], ddx_ref[:, i] = generate_sin_trajectory(t, x0, amp, phi, freq)
        v[:, i] = np.squeeze(simu.v)
        q[:, i] = np.squeeze(simu.q)

        try:
            # Calcolo della dinamica
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

            if test['controllore'] == 'OSC':
                tau[:, i] = np.squeeze(J_full.T @ f) + np.squeeze(NJ @ (20 * tau_0 + h))
                #tau[:, i] = np.squeeze(J_full.T @ f) + np.squeeze(NJ @ (10 * tau_0 + h))
            elif test['controllore'] == 'IC':
                tau[:, i] = np.squeeze(h + J_full.T @ (8 * ddx_fb) + NJ @ tau_0)
            else:
                print('ERROR: Unknown controller', test['controllore'])
                sys.exit(0)

            if test['metodo_integrazione'] == 'Euler':
                simu.simulate(tau[:, i], test['dt_simulation'], N_simulation)
            elif test['metodo_integrazione'] == 'RK4':
                simu.simulate(tau[:, i], test['dt_simulation'], N_simulation)
            else:
                print('ERROR: Unknown integration method', test['metodo_integrazione'])
                sys.exit(0)
            
            tau_c[:, i] = simu.tau_c
            ddx[:, i] = np.array(J_full @ simu.dv + dJdq).flatten()
            t += conf['dt_control']

            if end_effector_position[2] < z_superficie and i > 1000:
                ddq = simu.get_ddq(tau[:, i])
                ddx_actual = np.array(J_full @ ddq).flatten()
                if np.any(np.isnan(ddx_actual)):
                    print(f"NaN detected in end-effector accelerations at step {i}: {ddx_actual}")
                    instabile = True
                    break
            accelerazione_ee_xy[:, i] = ddx_actual[:2]

            velocita.append(np.linalg.norm(simu.v))
            
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
        
            

    
    # Calcoli aggiuntivi per il tracciamento e visualizzazioni
    inf_norm_error = np.linalg.norm(x_ref - x, ord=np.inf, axis=0)
    tracking_err = np.sum(norm(x_ref - x, axis=0)) / N_control
    desc = test['controllore'] + ' kp=' + str(test['kp']) + ' k_attrito =' + str(test['k_attrito'])

    if test['controllore'] == 'OSC' and test['metodo_integrazione'] == 'RK4':
        tracking_err_osc_RK4.append({'value': tracking_err, 'description': desc})
        inf_norm_errors_osc_RK4.append({'value': inf_norm_error, 'description': desc})
        dt_simulation_values.append(test['dt_simulation'])
    elif test['controllore'] == 'OSC' and test['metodo_integrazione'] == 'Euler':
        tracking_err_osc_Euler.append({'value': tracking_err, 'description': desc})
        inf_norm_errors_osc_Euler.append({'value': inf_norm_error, 'description': desc})
        dt_simulation_values.append(test['dt_simulation'])
    elif test['controllore'] == 'IC' and test['metodo_integrazione'] == 'RK4':
        tracking_err_ic_RK4.append({'value': tracking_err, 'description': desc})
        inf_norm_errors_ic_RK4.append({'value': inf_norm_error, 'description': desc})
        dt_simulation_values.append(test['dt_simulation'])
    elif test['controllore'] == 'IC' and test['metodo_integrazione'] == 'Euler':
        tracking_err_ic_Euler.append({'value': tracking_err, 'description': desc})
        inf_norm_errors_ic_Euler.append({'value': inf_norm_error, 'description': desc})
        dt_simulation_values.append(test['dt_simulation'])
    else:
        print('ERROR: Unknown controller', test['controllore'])
        sys.exit(0)
    
    print('Average tracking error %.9f m\n' % (tracking_err))
    print('Average inf norm error %.9f m\n' % (np.mean(inf_norm_error)))

    # Determina l'intervallo di dati validi in base allo stato della simulazione
    if instabile:
        valid_range = range(i)
    else:
        valid_range = range(N_control)

    valid_indices = list(valid_range)

    valid_velocities = np.linalg.norm(v[:, valid_indices], axis=0)
    valid_velocities = valid_velocities[~np.isnan(valid_velocities)]

    valid_accelerations = np.linalg.norm(accelerazione_ee_xy[:, valid_indices], axis=0)
    valid_accelerations = valid_accelerations[~np.isnan(valid_accelerations)]

    # Calcolo delle statistiche con i dati validi
    if valid_velocities.size > 0:
        velocita_media = np.mean(valid_velocities)
        std_velocita = np.std(valid_velocities)
        max_velocita = np.max(valid_velocities)
        min_velocita = np.min(valid_velocities)
    else:
        velocita_media = 0
        std_velocita = 0
        max_velocita = 0
        min_velocita = 0

    if valid_accelerations.size > 0:
        accelerazione_media_ee_xy = np.mean(valid_accelerations)
        std_accelerazione_ee_xy = np.std(valid_accelerations)
        max_accelerazione_ee_xy = np.max(valid_accelerations)
        min_accelerazione_ee_xy = np.min(valid_accelerations)
    else:
        accelerazione_media_ee_xy = 0
        std_accelerazione_ee_xy = 0
        max_accelerazione_ee_xy = 0
        min_accelerazione_ee_xy = 0



    # Stampa delle statistiche calcolate con messaggi aggiornati
    print(f'Velocità media dei giunti: {velocita_media:.3f} rad/s')

    # Aggiornato per specificare che l'accelerazione è durante il contatto
    print(f'Accelerazione media dell\'EE (xy) durante il contatto con la superficie: {accelerazione_media_ee_xy:.3f} m/s^2')

    print(f'Deviazione standard della velocità dei giunti: {std_velocita:.3f} rad/s')

    # Aggiornato per specificare che la deviazione standard dell'accelerazione è durante il contatto
    print(f'Deviazione standard dell\'accelerazione dell\'EE (xy) durante il contatto: {std_accelerazione_ee_xy:.3f} m/s^2')

    print(f'Velocità massima dei giunti: {max_velocita:.3f} rad/s')
    print(f'Velocità minima dei giunti: {min_velocita:.3f} rad/s')

    # Aggiornato per specificare che l'accelerazione massima e minima sono durante il contatto
    print(f'Accelerazione massima dell\'EE (xy) durante il contatto: {max_accelerazione_ee_xy:.3f} m/s^2')
    print(f'Accelerazione minima dell\'EE (xy) durante il contatto: {min_accelerazione_ee_xy:.3f} m/s^2')
    

    (f, ax) = plt.subplots(num_dof, 1, figsize=(10, 20))
    tt = np.arange(0.0, N_control * conf['dt_control'], conf['dt_control'])

    for i in range(num_dof):
        ax[i].plot(tt, q[i, :N_control], label=f'q_{i}')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(f'q_{i} [rad]')
        ax[i].legend()
    plt.suptitle(f'Traiettorie dei Giunti per k_attrito = {test["k_attrito"]}, del controllore {test["controllore"]} con un time step di {test["dt_simulation"]} s')
    plt.show()

    plt.figure()
    for i in range(nx):
        plt.plot(tt, x[i, :N_control], label=f'x_{i}')
        plt.plot(tt, x_ref[i, :N_control], '--', label=f'x_ref_{i}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title(f'Traiettorie Cartesiane per k_attrito = {test["k_attrito"]}, del controllore {test["controllore"]} con un time step di {test["dt_simulation"]} s')
    plt.legend()
    plt.show()
