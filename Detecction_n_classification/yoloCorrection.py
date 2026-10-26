import os
import glob


dataset_folders = [
    "YOLODataset-Camera_Free-11-nolisto",
    "YOLODataset-Camera_Free2-nolisto",    
]

total_archivos_modificados = 0

for dataset_name in dataset_folders:
    labels_folder = os.path.join(dataset_name, "labels")
    
    # Validar que la carpeta de labels exista
    if not os.path.exists(labels_folder):
        print(f"Advertencia: No se encontró la carpeta '{labels_folder}'. Saltando...")
        continue

    # Buscar todos los archivos .txt dentro de la carpeta labels actual
    txt_files = glob.glob(os.path.join(labels_folder, "*.txt"))
    print(f"Procesando {len(txt_files)} etiquetas en '{labels_folder}'...")

    for file_path in txt_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # Validar que la línea tenga el formato correcto de YOLO (5 valores)
            if len(parts) == 5:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Aplicar la corrección del eje Y invertido
                new_y_center = 1.0 - y_center
                
                # Formatear la nueva línea manteniendo los 6 decimales
                new_line = f"{class_id} {x_center:.6f} {new_y_center:.6f} {w:.6f} {h:.6f}\n"
                new_lines.append(new_line)
                
        # Sobreescribir el archivo original con las coordenadas corregidas
        with open(file_path, 'w') as file:
            file.writelines(new_lines)
            
        total_archivos_modificados += 1

print(f"\n¡Conversión masiva finalizada! Se corrigieron {total_archivos_modificados} archivos .txt en total.")