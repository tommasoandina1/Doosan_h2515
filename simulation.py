import numpy as np
from numpy import nan
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time, sys


from function import Simulator, z_superficie, generate_sin_trajectory

# Configuration
urdf_path = "/Users/tommasoandina/Desktop/Doosan_h2515-main/model.urdf"
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


tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/2000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}]

tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]
tests += [{'controllore': 'OSC', 'kp': 1000, 'dt_simulation': 1/1000, 'k_attrito': 0.15, 'metodo_integrazione': 'Euler'}]






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



for (test_id, test) in  enumerate(tests):
    print()
    description = f"{test['controllore']} kp={test['kp']} dt_simulation={test['dt_simulation']} k_attrito={test['k_attrito']} metodo={test['metodo_integrazione']}"
    print(description)
    simu = Simulator(num_dof, mass_matrix_fun, bias_force_fun, jacobian_fun, forward_kinematics_fun, conf['T_SIMULATION'], test['metodo_integrazione'])
    kp = test['kp']
    kd = 2 * np.sqrt(kp)
    k_attrito = test['k_attrito']
    simu.set_k_attrito(k_attrito)
    simu.init(conf['q0'])
    dt_simulation_values.append(test['dt_simulation'])
    metodo_integrazione=test['metodo_integrazione']
    



    accelerazioni = []
    velocita = []
    instabile = False
    bloccato = False
    
    
    nx, ndx = 3, 3
    N_control = int(conf['T_SIMULATION'] / conf['dt_control'])
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

    x0          = np.array([0.032, 0.09, 1]).T          # offset
    #amp         = np.array([0.3, 0.5, 0.15]).T          # amplitude
    amp         = np.array([0.3, 0.5, 0.013]).T
    phi         = np.array([0.0, 0, 0]).T               # phase
    freq        = np.array([0.5, 0.5, 0.5]).T           

    
    

    t = 0.0
    PRINT_N = int(conf['PRINT_T'] / conf['dt_control'])

    
    for i in range(0, N_control):
        time_start = time.time()
        ddx_actual = np.zeros(3)

        # set reference trajectory
        x_ref[:, i], dx_ref[:, i], ddx_ref[:, i] = generate_sin_trajectory(t, x0, amp, phi, freq)
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

        

            if(test['controllore']=='OSC'):
                tau[:, i] = np.squeeze(J_full.T @ f) + np.squeeze(NJ @ (tau_0 + h))
            elif(test['controllore']=='IC'):
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
            ddx[:, i]= np.array(J_full @ simu.dv + dJdq).flatten()
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



    inf_norm_error = np.linalg.norm(x_ref - x, ord=np.inf, axis=0)
    tracking_err = np.sum(norm(x_ref - x, axis=0)) / N_control
    desc = test['controllore']+' kp='+str(test['kp'])+' k_attrito ='+str(test['k_attrito'])

    if(test['controllore']=='OSC' and test['metodo_integrazione'] == 'RK4'):        
        tracking_err_osc_RK4 += [{'value': tracking_err, 'description': desc}]
        inf_norm_errors_osc_RK4 += [{'value': inf_norm_error, 'description': desc}]
    elif(test['controllore']=='IC' and test['metodo_integrazione'] == 'RK4'):
        tracking_err_ic_RK4 += [{'value': tracking_err, 'description': desc}]
        inf_norm_errors_ic_RK4 += [{'value': inf_norm_error, 'description': desc}]
    elif(test['controllore']=='IC' and test['metodo_integrazione'] == 'Euler'):
        tracking_err_ic_Euler += [{'value': tracking_err, 'description': desc}]
        inf_norm_errors_ic_Euler += [{'value': inf_norm_error, 'description': desc}]
    elif(test['controllore']=='OSC' and test['metodo_integrazione'] == 'Euler'):
        tracking_err_osc_Euler += [{'value': tracking_err, 'description': desc}]
        inf_norm_errors_osc_Euler += [{'value': inf_norm_error, 'description': desc}]
    else:
        print('ERROR: Unknown controller', test['controllore'])
    
    print('Average tracking error %.5f m\n'%(tracking_err))
    print('Average inf norm error %.5f m\n'%(np.mean(inf_norm_error)))


   
    # Determina l'intervallo di dati validi in base allo stato della simulazione
    if instabile:
        # Se la simulazione è instabile, utilizza solo i dati raccolti fino a `i`
        valid_range = range(i)
    else:
        # Se la simulazione non è instabile, utilizza tutti i dati raccolti
        valid_range = range(N_control)

    # Converti l'intervallo in una lista di indici per una gestione più flessibile
    valid_indices = list(valid_range)

    # Rimuovere i NaN dai dati validi
    valid_velocities = np.linalg.norm(v[:, valid_indices], axis=0)
    valid_velocities = valid_velocities[~np.isnan(valid_velocities)]

    valid_accelerations = np.linalg.norm(accelerazione_ee_xy[:, valid_indices], axis=0)
    valid_accelerations = valid_accelerations[~np.isnan(valid_accelerations)]

    # Calcolo delle statistiche con i dati validi, solo se i dati sono disponibili
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
        ax[i].plot(tt, q[i, :-1], label=f'q_{i}')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(f'q_{i} [rad]')
        ax[i].legend()
    plt.suptitle(f'Traiettorie dei Giunti per k_attrito = {test['k_attrito']}, del controllore {test["controllore"]} con un time step di {test["dt_simulation"]} s')
    plt.show()
    
    # Plot delle Traiettorie Cartesiane
    plt.figure()
    for i in range(nx):
        plt.plot(tt, x[i, :], label=f'x_{i}')
        plt.plot(tt, x_ref[i, :], '--', label=f'x_ref_{i}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title(f'Traiettorie Cartesiane per k_attrito = {test['k_attrito']}, del controllore {test["controllore"]} con un time step di {test["dt_simulation"]} s')
    plt.legend()
    plt.show()



    start_index = 1000

    # Assicurati che l'indice di partenza sia valido
    if N_control > start_index:
        # Crea un vettore di tempo ridotto per il plot che parte da start_index
        tt_reduced = np.arange(start_index * conf['dt_control'], N_control * conf['dt_control'], conf['dt_control'])
        
        # Riduci accelerazione_ee_xy per plottare solo da start_index in poi
        accelerazione_ee_xy_reduced = accelerazione_ee_xy[:, start_index:]

        # Plot delle Accelerazioni dell'End-Effector (xy) dopo start_index passi
        plt.figure(figsize=(10, 6))
        plt.plot(tt_reduced, accelerazione_ee_xy_reduced[0, :], label='Acc. EE in x')
        plt.plot(tt_reduced, accelerazione_ee_xy_reduced[1, :], label='Acc. EE in y')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.title(f'Accelerazioni dell\'EE per k_attrito = {test["k_attrito"]}, del controllore {test['controllore']} con un time step di {test["dt_simulation"]} s')
        plt.legend()
        plt.show()
    

# Estrai i valori delle norme inf dagli errori
inf_norm_errors_osc_RK4_values = [np.mean(item['value']) for item in inf_norm_errors_osc_RK4]
inf_norm_errors_osc_Euler_values = [np.mean(item['value']) for item in inf_norm_errors_osc_Euler]
inf_norm_errors_ic_RK4_values = [np.mean(item['value']) for item in inf_norm_errors_ic_RK4]
inf_norm_errors_ic_Euler_values = [np.mean(item['value']) for item in inf_norm_errors_ic_Euler]

# Verifica la lunghezza delle liste
print(f"Lunghezza di dt_simulation_values: {len(dt_simulation_values)}")
print(f"Lunghezza di inf_norm_errors_osc_RK4_values: {len(inf_norm_errors_osc_RK4_values)}")
print(f"Lunghezza di inf_norm_errors_osc_Euler_values: {len(inf_norm_errors_osc_Euler_values)}")
print(f"Lunghezza di inf_norm_errors_ic_RK4_values: {len(inf_norm_errors_ic_RK4_values)}")
print(f"Lunghezza di inf_norm_errors_ic_Euler_values: {len(inf_norm_errors_ic_Euler_values)}")


# Plot delle curve
plt.figure(figsize=(10, 6))
plt.plot(dt_simulation_values, inf_norm_errors_osc_RK4_values, 'o-', label='OSC con RK4')
plt.plot(dt_simulation_values, inf_norm_errors_osc_Euler_values, 's-', label='OSC con Euler')
plt.plot(dt_simulation_values, inf_norm_errors_ic_RK4_values, '^-', label='IC con RK4')
plt.plot(dt_simulation_values, inf_norm_errors_ic_Euler_values, 'd-', label='IC con Euler')
plt.xlabel('dt_simulation [s]')
plt.ylabel('Norma Inf Errore Posizione [m]')
plt.title('Norma Inf Errore di Posizione dell\'EE per Metodo di Integrazione e Controllore')
plt.legend()
plt.show()
