import numpy as np
from numpy import nan
import casadi as cs
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
import time, sys
from numpy.linalg import norm

from function import Simulator

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

# Simulation parameters
conf = {
    'T_SIMULATION': 6.0,
    'dt_control': 1/1000,
    'PRINT_T': 1.0,
    'simulate_real_time': False,
    'kp_j': 100.0,
    'kd_j': np.sqrt(100),
    'kp': 1000,
    'kd': np.sqrt(1000),
    'q0': np.zeros(num_dof),
}

# Define the test configurations
tests = [
    {'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/32000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'},
    {'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'},
    {'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/8000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'},
    {'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/4000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'},

]

# Simulazione di "ground truth" per riferimento
ground_truth_test = {'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/64000, 'k_attrito': 0.15, 'metodo_integrazione': 'RK4'}
simu_ground_truth = Simulator(num_dof, mass_matrix_fun, bias_force_fun, jacobian_fun, forward_kinematics_fun, conf['T_SIMULATION'], ground_truth_test['metodo_integrazione'])
simu_ground_truth.set_k_attrito(ground_truth_test['k_attrito'])
simu_ground_truth.init(conf['q0'])

N_control = int(conf['T_SIMULATION'] / conf['dt_control'])
N_simulation_gt = int(conf['dt_control'] / ground_truth_test['dt_simulation'])

# Initialize ground truth data arrays
q_gt = np.full((num_dof, N_control + 1), nan)
v_gt = np.full((num_dof, N_control + 1), nan)

# Run ground truth simulation
for i in range(N_control):
    q_gt[:, i] = np.squeeze(simu_ground_truth.q)
    v_gt[:, i] = np.squeeze(simu_ground_truth.v)
    tau_gt = np.zeros(num_dof)  # Placeholder for tau_gt computation

    simu_ground_truth.simulate(tau_gt, ground_truth_test['dt_simulation'], N_simulation_gt)
    q_gt[:, i+1] = simu_ground_truth.q.flatten()
    v_gt[:, i+1] = simu_ground_truth.v.flatten()

# Initialize error storage
errors = []

# Run simulations and calculate accuracy for each test configuration
for (test_id, test) in enumerate(tests):
    print(f"\nRunning test {test_id+1}: {test['controllore']} with dt_simulation={test['dt_simulation']}")

    simu = Simulator(num_dof, mass_matrix_fun, bias_force_fun, jacobian_fun, forward_kinematics_fun, conf['T_SIMULATION'], test['metodo_integrazione'])
    simu.set_k_attrito(test['k_attrito'])
    simu.init(conf['q0'])

    N_simulation = int(conf['dt_control'] / test['dt_simulation'])
    
    q = np.full((num_dof, N_control + 1), nan)
    v = np.full((num_dof, N_control + 1), nan)
    
    local_errors = []

    for i in range(N_control):
        q[:, i] = np.squeeze(simu.q)
        v[:, i] = np.squeeze(simu.v)

        tau = np.zeros(num_dof)  # Placeholder for tau computation

        simu.simulate(tau, test['dt_simulation'], N_simulation)
        q[:, i+1] = simu.q.flatten()
        v[:, i+1] = simu.v.flatten()

        # Calculate local integration error
        q_error_local = np.linalg.norm(q[:, i+1] - q_gt[:, i+1], ord=np.inf)
        local_errors.append(q_error_local)

    mean_local_error = np.mean(local_errors)

    errors.append({
        'description': f"{test['controllore']} with dt_simulation={test['dt_simulation']}",
        'mean_local_error': mean_local_error
    })

# Plot results for error analysis
time_steps = [test['dt_simulation'] for test in tests]
mean_errors = [error['mean_local_error'] for error in errors]

plt.figure()
plt.plot(time_steps, mean_errors, '-o', label='Mean Error Norm 2')
plt.xscale('log')
plt.xlabel('Time Step [s]')
plt.ylabel('Mean Error Norm 2')
plt.title('Mean Error Norm 2 vs Time Step')
plt.legend()
plt.show()
