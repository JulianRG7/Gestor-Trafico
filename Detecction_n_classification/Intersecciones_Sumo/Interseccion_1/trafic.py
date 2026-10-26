import traci
import socket

# Configuración de red para hablar con Unity
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Iniciar SUMO (usa 'sumo' en lugar de 'sumo-gui' para que sea más rápido)
traci.start(["sumo", "-c", "Prueba1.sumocfg"]) 

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    veh_ids = traci.vehicle.getIDList()
    mensaje = ""
    
    for v_id in veh_ids:
        x, y = traci.vehicle.getPosition(v_id)
        angle = traci.vehicle.getAngle(v_id)
        v_type = traci.vehicle.getTypeID(v_id) # auto, ambulancia, etc.
        
        # Formato: id,x,y,angulo,tipo;
        mensaje += f"{v_id},{x},{y},{angle},{v_type};"
    
    # Enviar a Unity
    if mensaje:
        sock.sendto(mensaje.encode(), (UDP_IP, UDP_PORT))

traci.close()

