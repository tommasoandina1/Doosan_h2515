import pinocchio as pin
import numpy as np
import meshcat
import time
from numpy.linalg import norm
from function import Simulator, generate_sin_trajectory

# Percorso del file URDF
urdf_path = "/Users/tommasoandina/Desktop/doosan-robot2-master/dsr_description2/urdf/h2515.white.urdf"



# Definisci l'end-effector (l'ultimo link del robot)
end_effector = 'link6'

# Carica il modello URDF con Pinocchio
try:
    rmodel, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    print("URDF caricato con successo con Pinocchio.")
except Exception as e:
    print(f"Errore nel caricamento del file URDF per Pinocchio: {e}")
    exit()

# Crea il contesto cinematico per il modello Pinocchio
data = rmodel.createData()

# Inizializza il visualizzatore Meshcat
viz = pin.visualize.MeshcatVisualizer(rmodel, collision_model, visual_model)
viz.initViewer(loadModel=True, open=True)
viz.loadViewerModel()

# Mostra il robot con i giunti tutti a zero
viz.display(np.zeros(rmodel.nq))

# Parametri di simulazione
conf = {
    'T_SIMULATION': 6.0,
    'dt_control': 1/1000,
    'PRINT_T': 1.0,
    'simulate_real_time': False,
    'kp_j': 100.0,
    'kd_j': np.sqrt(100),
    'kp': 1000,
    'kd': np.sqrt(1000),
    'q0': np.zeros(rmodel.nq),
}

# Test di controllo
tests = []
tests += [{'controllore': 'IC', 'kp': 1000, 'dt_simulation': 1/16000, 'k_attrito': 0.0, 'metodo_integrazione': 'RK4'}]

N_control = int(conf['T_SIMULATION'] / conf['dt_control'])
for (test_id, test) in enumerate(tests):
    print(f"Simulazione: {test['controllore']} kp={test['kp']} dt_simulation={test['dt_simulation']} k_attrito={test['k_attrito']} metodo={test['metodo_integrazione']}")

    # Inizializza il simulatore (Pinocchio)
    q_log = []  # Per salvare le configurazioni dei giunti durante la simulazione
    q = conf['q0']  # Configurazione iniziale
    dq = np.zeros(rmodel.nv)  # Velocità iniziali
    t = 0.0  # Tempo di simulazione

    for i in range(N_control):
        t += conf['dt_control']

        # Genera traiettoria di riferimento
        x0 = np.array([0.032, 0.09, 1]).T
        amp = np.array([0.3, 0.5, 0.15]).T
        phi = np.array([0.0, 0, 0]).T
        freq = np.array([0.5, 0.5, 0.5]).T
        x_ref, dx_ref, ddx_ref = generate_sin_trajectory(t, x0, amp, phi, freq)

        # Calcola la cinematica diretta con Pinocchio
        pin.forwardKinematics(rmodel, data, q, dq)
        pin.updateFramePlacements(rmodel, data)  # Aggiorna la posizione degli elementi

        # Calcolo dinamica con Pinocchio
        M = pin.crba(rmodel, data, q)  # Matrice di massa
        b = pin.rnea(rmodel, data, q, dq, np.zeros(rmodel.nv))  # Forze di bias
        

        J = pin.computeFrameJacobian(rmodel, data, q, rmodel.getFrameId(end_effector), pin.LOCAL_WORLD_ALIGNED)
        J_linear = J[:3, :]  
        

        # Legge la posizione dell'end-effector
        x_ee = data.oMf[rmodel.getFrameId(end_effector)].translation

        # Controllo di posizione
        ddx_fb = conf['kp'] * (x_ref - x_ee) + conf['kd'] * (dx_ref - J_linear @ dq)

        f = ddx_fb  

       
        tau = np.zeros(rmodel.nv)
        
        tau += J_linear.T @ f  # Moltiplicazione matrice-vettore
        
        tau += b

        dq += conf['dt_control'] * np.linalg.inv(M) @ (tau - b)
        q += conf['dt_control'] * dq

        # Salva la configurazione attuale per la visualizzazione
        q_log.append(q.copy())

        # Se `simulate_real_time` è attivo, aggiungi un delay
        if conf['simulate_real_time']:
            time.sleep(conf['dt_control'])

    # Alla fine della simulazione, visualizza la traiettoria salvata
    for q_step in q_log:
        viz.display(q_step)  # Visualizza ogni posizione dei giunti
        time.sleep(0.05)  # Pausa per l'animazione
