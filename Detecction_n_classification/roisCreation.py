import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================
ALL_SOURCES = [
    Path("YOLODataset-Camera_Free3-L-V1"),
    Path("YOLODataset-Camera_Free3-L-V2"),
    Path("YOLODataset"),
    Path("YOLODataset-Camera_Free-V1"),
    Path("YOLODataset-Camera_Free2"),
    Path("YOLODataset-Camera_Free-11"),
    Path("YOLODataset-Camera_Free2-V2")
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_JSON = "rois.json"

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def get_first_image(source_path):
    images_dir = source_path / "images"
    if not images_dir.exists():
        return None
    for img_path in images_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(img_path)
    return None

def main():
    global points
    roi_dict = {}

    print("=== CREADOR DE ROIs PARA YOLO ===")
    print("Controles:")
    print(" - Clic izquierdo: Agregar punto al polígono")
    print(" - 'c': Limpiar puntos actuales")
    print(" - 's': Guardar ROI y pasar a la siguiente carpeta")
    print(" - 'q': Salir sin guardar el progreso actual\n")

    for source in ALL_SOURCES:
        img_path = get_first_image(source)
        if not img_path:
            print(f"AVISO: No se encontraron imágenes en {source}/images. Saltando...")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error al leer {img_path}. Saltando...")
            continue

        clone = img.copy()
        h, w = img.shape[:2]
        points = []
        window_name = f"Definiendo ROI: {source.name}"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            temp_img = clone.copy()
            
            # Dibujar el polígono en tiempo real
            if len(points) > 0:
                cv2.polylines(temp_img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
                for p in points:
                    cv2.circle(temp_img, p, 4, (0, 0, 255), -1)

            cv2.imshow(window_name, temp_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                points = []
                print("Puntos limpiados.")
            elif key == ord('s'):
                if len(points) >= 3:
                    # Normalizar coordenadas (0 a 1) para que sean independientes de la resolución
                    normalized_points = [[px / w, py / h] for px, py in points]
                    roi_dict[source.name] = normalized_points
                    print(f"ROI guardado para {source.name}")
                    break
                else:
                    print("Un polígono necesita al menos 3 puntos.")
            elif key == ord('q'):
                print("Operación cancelada por el usuario.")
                cv2.destroyAllWindows()
                return

        cv2.destroyWindow(window_name)

    # Guardar en JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(roi_dict, f, indent=4)
    print(f"\n¡Proceso terminado! Archivo generado: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

# import cv2
# import numpy as np
# import sys

# def main():
#     # 1. Definir el pixel exacto donde se hará el corte (eje X)
#     punto_corte_x = 1000  # <--- Cambia este número por el pixel que necesites

#     # 2. Rutas de tus imágenes
#     ruta_img1 = 'image_000002.png' 
#     ruta_img2 = 'image_000993.png' 

#     img1 = cv2.imread(ruta_img1)
#     img2 = cv2.imread(ruta_img2)

#     if img1 is None or img2 is None:
#         print("Error: No se pudieron cargar las imágenes. Verifica las rutas.")
#         sys.exit()

#     # 3. Igualar tamaños (redimensionamos la img2 al tamaño de la img1)
#     alto1, ancho1 = img1.shape[:2]
#     img2 = cv2.resize(img2, (ancho1, alto1))

#     # Validar que el punto de corte esté dentro de los límites de la imagen
#     if punto_corte_x < 0 or punto_corte_x > ancho1:
#         print(f"Error: El pixel de corte ({punto_corte_x}) está fuera del ancho de la imagen ({ancho1} px).")
#         sys.exit()

#     # 4. Recortar y unir las imágenes
#     # Lado izquierdo de la imagen 1 (desde 0 hasta punto_corte_x)
#     mitad_izquierda = img1[:, :punto_corte_x]
    
#     # Lado derecho de la imagen 2 (desde punto_corte_x hasta el final)
#     mitad_derecha = img2[:, punto_corte_x:]

#     # Concatenar horizontalmente usando NumPy
#     imagen_final = np.hstack((mitad_izquierda, mitad_derecha))

#     # 5. Mostrar y guardar el resultado
#     cv2.imshow('Imagen Final Fusionada', imagen_final)
#     print("Presiona cualquier tecla sobre la imagen para guardarla y cerrar.")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     cv2.imwrite('resultado_fusion.jpg', imagen_final)
#     print("Imagen guardada exitosamente como 'resultado_fusion.jpg'")

# if __name__ == "__main__":
#     main()