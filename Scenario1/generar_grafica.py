import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Leer el archivo (Busca el SOTA primero, si no, busca el normal)
archivo = "validacion_trafico_sota.csv"
if not os.path.exists(archivo):
    archivo = "validacion_trafico.csv"
    if not os.path.exists(archivo):
        print("Error: No se encontró ningún archivo CSV. Ejecuta Fuzzy.py primero.")
        exit()

df = pd.read_csv(archivo)

# 2. Agrupar de forma dinámica (Toma el nombre de la 1ra columna automáticamente)
col_estado = df.columns[0]
resumen = df.groupby(col_estado).mean()

# Verificar que existan pruebas de ambos (True y False)
if True not in resumen.index or False not in resumen.index:
    print("Aviso: Necesitas ejecutar el script tanto con IA (True) como sin IA (False).")
    print("Sugerencia: Usa el menú 'Opción 2' de tu Fuzzy.py para generar los datos rápido.")
    exit()

estatico = resumen.loc[False]
inteligente = resumen.loc[True]

# 3. Extraer los datos de las calles (dinámicamente desde las columnas 0 a 3 del resumen)
cols_calles = resumen.columns[0:4] 
valores_estatico = [estatico[col] for col in cols_calles]
valores_inteligente = [inteligente[col] for col in cols_calles]

# Nombres limpios para la gráfica
nombres_ejes = ['Norte', 'Sur', 'Este', 'Oeste']

# 4. Dibujar la Gráfica
x = np.arange(len(nombres_ejes))
ancho_barra = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

barras1 = ax.bar(x - ancho_barra/2, valores_estatico, ancho_barra, label='Semáforo Estático', color='#e74c3c')
barras2 = ax.bar(x + ancho_barra/2, valores_inteligente, ancho_barra, label='Sistema Fuzzy/YOLO', color='#2ecc71')

ax.set_ylabel('Promedio de Autos Detenidos')
ax.set_title('Impacto del Sistema de Inteligencia Artificial (SOTA)', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(nombres_ejes)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Etiquetas numéricas sobre las barras
def autolabel(rects):
    for rect in rects:
        alto = rect.get_height()
        ax.annotate(f'{alto:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, alto),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(barras1)
autolabel(barras2)

# 5. Cálculo del Total (la última columna)
col_total = resumen.columns[4]
total_estatico = estatico[col_total]
total_inteligente = inteligente[col_total]
mejora_porcentaje = ((total_estatico - total_inteligente) / total_estatico) * 100

texto_conclusion = f'Tráfico Total Estático: {total_estatico:.1f} autos  |  Tráfico Total IA: {total_inteligente:.1f} autos\n¡Reducción global de la congestión del {mejora_porcentaje:.1f}%!'

plt.figtext(0.5, 0.02, texto_conclusion, ha="center", fontsize=11, 
            bbox={"facecolor":"#d4edda", "alpha":0.8, "pad":5, "edgecolor":"#c3e6cb"})

plt.subplots_adjust(bottom=0.15) 
plt.savefig('grafica_sota_final.png', dpi=300)
print(f"¡Gráfica generada con éxito! Revisa la imagen 'grafica_sota_final.png'.")
plt.show()