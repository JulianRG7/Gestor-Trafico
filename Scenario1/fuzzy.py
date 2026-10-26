import os, sys, time, subprocess
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- CONFIGURACIÓN ---
USAR_FUZZY = False  # Cambia a False para comparar con el semáforo tradicional
ARCHIVO_LOG = "validacion_trafico_sota.csv"

# --- 1. SETUP FUZZY OMNIDIRECCIONAL (SOTA) ---
dens_v = ctrl.Antecedent(np.arange(0, 101, 1), 'densidad_verde')
dens_r = ctrl.Antecedent(np.arange(0, 101, 1), 'densidad_roja')
t_rojo = ctrl.Antecedent(np.arange(0, 121, 1), 'tiempo_rojo')
extension = ctrl.Consequent(np.arange(0, 16, 1), 'extension', defuzzify_method='centroid')

# --- 2. MEMBRESÍAS OPTIMIZADAS ---
dens_v['baja']  = fuzz.trapmf(dens_v.universe, [0, 0, 15, 30])
dens_v['media'] = fuzz.trimf(dens_v.universe,  [20, 40, 70])
dens_v['alta']  = fuzz.trapmf(dens_v.universe, [60, 80, 100, 100])

dens_r['baja']  = fuzz.trapmf(dens_r.universe, [0, 0, 15, 30])
dens_r['media'] = fuzz.trimf(dens_r.universe,  [20, 40, 70])
dens_r['alta']  = fuzz.trapmf(dens_r.universe, [60, 80, 100, 100])

t_rojo['corto'] = fuzz.trapmf(t_rojo.universe, [0, 0, 20, 45])
t_rojo['largo'] = fuzz.trapmf(t_rojo.universe, [35, 60, 120, 120])

extension['nula']  = fuzz.trimf(extension.universe, [0,  0,  0])
extension['corta'] = fuzz.trimf(extension.universe, [5,  5,  5])
extension['media'] = fuzz.trimf(extension.universe, [10, 10, 10])
extension['larga'] = fuzz.trimf(extension.universe, [15, 15, 15])

# --- 3. REGLAS ---
rules = [
    ctrl.Rule(dens_v['baja'], extension['nula']),
    ctrl.Rule(dens_v['media'] & dens_r['baja']  & t_rojo['corto'], extension['media']),
    ctrl.Rule(dens_v['media'] & dens_r['baja']  & t_rojo['largo'], extension['corta']),
    ctrl.Rule(dens_v['media'] & dens_r['media'] & t_rojo['corto'], extension['corta']),
    ctrl.Rule(dens_v['media'] & dens_r['media'] & t_rojo['largo'], extension['nula']),
    ctrl.Rule(dens_v['media'] & dens_r['alta'],                     extension['nula']),
    ctrl.Rule(dens_v['alta']  & dens_r['baja']  & t_rojo['corto'], extension['larga']),
    ctrl.Rule(dens_v['alta']  & dens_r['baja']  & t_rojo['largo'], extension['media']),
    ctrl.Rule(dens_v['alta']  & dens_r['media'] & t_rojo['corto'], extension['media']),
    ctrl.Rule(dens_v['alta']  & dens_r['media'] & t_rojo['largo'], extension['corta']),
    ctrl.Rule(dens_v['alta']  & dens_r['alta']  & t_rojo['corto'], extension['media']),
    ctrl.Rule(dens_v['alta']  & dens_r['alta']  & t_rojo['largo'], extension['corta']),
]

sistema   = ctrl.ControlSystem(rules)
sim_fuzzy = ctrl.ControlSystemSimulation(sistema)

# --- 4. CONEXIÓN SUMO ---
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Error: Definir variable de entorno SUMO_HOME")
import traci

def ejecutar_simulacion():
    import zmq
    import json
    
    print("Iniciando SUMO y configurando servidor ZeroMQ para Unity...")
    
    # --- 1. LEVANTAR SERVIDORES ZEROMQ PARA UNITY ---
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556") # Por aquí enviaremos los autos y semáforos
    
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind("tcp://*:5557") # Por aquí escuchamos a Unity
    
    # --- 2. INICIAR SUMO ---
    traci.start(["sumo-gui", "-c", "Sumo2Unity.sumocfg", "--delay", "50"])
    
    id_tls = traci.trafficlight.getIDList()[0]
    
    CAPACIDAD_CARRIL = 25.0
    ID_CALLES = ["E0", "E3", "E5", "-E1"]
    
    stats = {calle: 0 for calle in ID_CALLES}
    paso = 0
    t_rojo_ns, t_rojo_f2, t_rojo_f4 = 0.0, 0.0, 0.0
    fase_prev = -1
    
    # Diccionario para rastrear los cambios de color de los semáforos
    last_tls_states = {}
    
    print(f"¡Todo listo! Ya puedes darle PLAY en Unity. USAR_FUZZY = {USAR_FUZZY}")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t_actual = traci.simulation.getTime()
        fase_actual = traci.trafficlight.getPhase(id_tls)
        
        # --- 3. TU LÓGICA DIFUSA (SOTA) ---
        for calle in ID_CALLES:
            stats[calle] += traci.edge.getLastStepHaltingNumber(calle)
        paso += 1

        if fase_actual != fase_prev:
            if fase_actual == 0: t_rojo_f2, t_rojo_f4 = t_actual, t_actual
            elif fase_actual == 2: t_rojo_ns, t_rojo_f4 = t_actual, t_actual
            elif fase_actual == 4: t_rojo_ns, t_rojo_f2 = t_actual, t_actual
            fase_prev = fase_actual

        if USAR_FUZZY and fase_actual in [0, 2, 4]:
            t_restante = traci.trafficlight.getNextSwitch(id_tls) - t_actual
            
            if 2.5 < t_restante < 3.5:
                if fase_actual == 0: 
                    autos_v = max(traci.edge.getLastStepVehicleNumber("E0"), traci.edge.getLastStepVehicleNumber("E3"))
                    autos_r = max(traci.edge.getLastStepVehicleNumber("E5"), traci.edge.getLastStepVehicleNumber("-E1"))
                    t_competidor = max(t_actual - t_rojo_f2, t_actual - t_rojo_f4)
                elif fase_actual == 2: 
                    autos_v = traci.edge.getLastStepVehicleNumber("E5")
                    autos_r = max(traci.edge.getLastStepVehicleNumber("E0"), traci.edge.getLastStepVehicleNumber("E3"), traci.edge.getLastStepVehicleNumber("-E1"))
                    t_competidor = max(t_actual - t_rojo_ns, t_actual - t_rojo_f4)
                elif fase_actual == 4: 
                    autos_v = traci.edge.getLastStepVehicleNumber("-E1")
                    autos_r = max(traci.edge.getLastStepVehicleNumber("E0"), traci.edge.getLastStepVehicleNumber("E3"), traci.edge.getLastStepVehicleNumber("E5"))
                    t_competidor = max(t_actual - t_rojo_ns, t_actual - t_rojo_f2)

                if autos_v == 0:
                    traci.trafficlight.setPhaseDuration(id_tls, 0)
                    print(f"Salto Inmediato: Eje vacío. Agilizando tráfico.")
                    continue

                dens_v_calc = min((autos_v / CAPACIDAD_CARRIL) * 100, 100)
                dens_r_calc = min((autos_r / CAPACIDAD_CARRIL) * 100, 100)
                
                sim_fuzzy.input['densidad_verde'] = dens_v_calc
                sim_fuzzy.input['densidad_roja'] = dens_r_calc
                sim_fuzzy.input['tiempo_rojo'] = min(t_competidor, 120)
                
                try:
                    sim_fuzzy.compute()
                    ext = round(sim_fuzzy.output['extension'])
                    if ext > 0:
                        nueva_dur = traci.trafficlight.getPhaseDuration(id_tls) + ext
                        traci.trafficlight.setPhaseDuration(id_tls, nueva_dur)
                        print(f"Fase {fase_actual} | V:{dens_v_calc:.0f}% R:{dens_r_calc:.0f}% | Wait:{t_competidor:.0f}s -> Ext: +{ext}s")
                except ValueError:
                    pass
        
        # --- 4. TRADUCTOR A UNITY (Anticolapso) ---
        # 4.1 Empaquetamos vehículos
        veh_list = []
        for v_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(v_id)
            veh_list.append({
                "vehicle_id": v_id,
                # Enviamos el eje Z como 0.0 explícitamente para evitar el IndexOutOfRange
                "position": [x, y, 0.0], 
                "angle": traci.vehicle.getAngle(v_id),
                "type": traci.vehicle.getTypeID(v_id),
                "long_speed": traci.vehicle.getSpeed(v_id),
                "vert_speed": 0.0,
                "lat_speed": 0.0
            })
        pub_socket.send_string(json.dumps({"type": "vehicles", "vehicles": veh_list}))

        # 4.2 Empaquetamos semáforos SOLO SI CAMBIARON DE COLOR
        tls_list = []
        for t_id in traci.trafficlight.getIDList():
            estado_actual = traci.trafficlight.getRedYellowGreenState(t_id)
            if last_tls_states.get(t_id) != estado_actual:
                tls_list.append({
                    "junction_id": t_id,
                    "state": estado_actual
                })
                last_tls_states[t_id] = estado_actual

        if len(tls_list) > 0:
            pub_socket.send_string(json.dumps({"type": "trafficlights", "lights": tls_list}))

        # 4.3 Drenamos el buzón de Unity COMPLETO para evitar fugas de memoria
        try:
            while True:
                router_socket.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            pass
        
        # 4.4 Freno de tiempo realista (aprox 20 FPS para dar respiro a Unity)
        time.sleep(0.05)

    traci.close()

    promedios = {calle: round(stats[calle] / paso, 2) for calle in ID_CALLES}
    suma_total = round(sum(promedios.values()), 2)
    
    existe_archivo = os.path.isfile(ARCHIVO_LOG)
    with open(ARCHIVO_LOG, "a") as f:
        if not existe_archivo:
            f.write("Fuzzy_Activo,Prom_Norte_E0,Prom_Sur_E3,Prom_Este_E5,Prom_Oeste_E1,Congestion_Total\n")
        f.write(f"{USAR_FUZZY},{promedios['E0']},{promedios['E3']},{promedios['E5']},{promedios['-E1']},{suma_total}\n")
    
    print(f"\n--- Análisis SOTA Finalizado ---")
    print(f"Congestión Promedio Total: {suma_total} autos detenidos/segundo.")

if __name__ == "__main__":
    ejecutar_simulacion()