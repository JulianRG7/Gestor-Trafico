import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# DATOS DEL REPORTE
# ==========================================
humedad = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

m1 = np.array([412.00, 296.45, 265.25, 213.71, 149.49, 118.93, 109.05, 107.41, 103.80, 101.78, 98.22])
m2 = np.array([412.88, 298.71, 265.33, 214.29, 150.58, 119.52, 109.52, 107.54, 103.62, 101.68, 98.78])
m3 = np.array([414.00, 295.80, 286.44, 206.28, 147.76, 114.96, 111.53, 106.46, 102.20, 101.66, 85.63])

repetibilidad = np.array([1.00, 1.53, 12.21, 4.47, 1.42, 2.48, 1.32, 0.59, 0.88, 0.06, 7.44])

promedio = (m1 + m2 + m3) / 3
std_dev = np.std([m1, m2, m3], axis=0)

# ==============================================================================
# Gráfica 1: REPETIBILIDAD # 1 (Líneas con Jitter y Textos)
# ==============================================================================
plt.figure(figsize=(14, 8))

# Paleta de colores morados
color_m1 = '#D8BFD8' 
color_m2 = '#BA55D3' 
color_m3 = '#663399' 

# TÉCNICA DE JITTER: Separamos las 3 líneas horizontalmente
offset = 1.5
h_m1 = humedad - offset
h_m2 = humedad
h_m3 = humedad + offset

plt.plot(h_m1, m1, marker='d', color=color_m1, label='M1', linewidth=1.5, markersize=8)
plt.plot(h_m2, m2, label='M2', marker='s', color=color_m2, linewidth=1.5, markersize=8)
plt.plot(h_m3, m3, marker='^', color=color_m3, label='M3', linewidth=1.5, markersize=8)

for i in range(len(humedad)):
    # Texto M1 (Abajo a la izquierda)
    plt.text(h_m1[i], m1[i] - 5, f'{m1[i]:.2f}', fontsize=8, ha='right', va='top', color='#8A6B8A', weight='bold')
    # Texto M2 (Arriba al centro)
    plt.text(h_m2[i], m2[i] + 7, f'{m2[i]:.2f}', fontsize=8, ha='center', va='bottom', color='#8B3A9E', weight='bold')
    # Texto M3 (Abajo a la derecha)
    plt.text(h_m3[i], m3[i] - 5, f'{m3[i]:.2f}', fontsize=8, ha='left', va='top', color=color_m3, weight='bold')

plt.xlabel('HUMEDAD (%)', fontsize=12)
plt.ylabel('SALIDA CRUDA (ADC)', fontsize=12)
plt.title('REPETIBILIDAD # 1\n(Comparativa de las 3 mediciones separadas)', fontsize=14, fontweight='bold')

plt.xticks(humedad)
# Límite superior ampliado para asegurar que la leyenda no estorbe
plt.ylim(50, 480) 

# LEYENDA UBICADA EN LA ESQUINA SUPERIOR DERECHA
plt.legend(loc='upper right', fontsize=10, frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('grafica_repetibilidad_1_final.png', dpi=300)
plt.show()

# ==============================================================================
# Gráfica 2: REPETIBILIDAD # 2 (Promedio con etiquetas de % de Repetibilidad)
# ==============================================================================
plt.figure(figsize=(10, 6))

line_color = '#4B0082' 

plt.errorbar(humedad, promedio, yerr=std_dev, fmt='o-', color=line_color, 
             ecolor=line_color, elinewidth=1.5, capsize=6, capthick=1.5, 
             markersize=8, label='Promedio')

for i in range(len(humedad)):
    offset_y = std_dev[i] + 12
    plt.text(humedad[i], promedio[i] - offset_y, f'{repetibilidad[i]}%', 
             fontsize=9, color=line_color, fontweight='bold', ha='center', va='top')

plt.xlabel('HUMEDAD (%)', fontsize=12)
plt.ylabel('SALIDA CRUDA (ADC)', fontsize=12)
plt.title('REPETIBILIDAD # 2\n(Promedio y Porcentaje de Repetibilidad)', fontsize=14, fontweight='bold')
plt.xticks(humedad)

# Añadimos la leyenda también a la esquina superior derecha por consistencia
plt.legend(loc='upper right', fontsize=10, frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('grafica_repetibilidad_2_final.png', dpi=300)
plt.show()