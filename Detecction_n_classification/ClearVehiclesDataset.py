import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")

# Definimos las clases según tu especificación
CLASES = {
    0: "Car",
    1: "Motorcycle",
    2: "Passenger",
    3: "Cargo",
    4: "Ambulance",
    5: "Police",
    6: "FireTruck"
}

class RevisorBoundingBoxes:
    def __init__(self, root):
        self.root = root
        self.root.title("Anotador YOLO/Unity - Dataset de Vehículos")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        self.carpeta_principal = None
        self.carpeta_images = None
        self.carpeta_labels = None

        self.lista_imagenes = []
        self.indice_actual = 0
        self.contador_borradas = 0

        self.imagen_tk = None
        self.imagen_pil = None
        
        # Variables para renderizado y matemáticas
        self.img_w = 0
        self.img_h = 0
        self.escala_actual = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Variables de estado de Bounding Boxes
        self.bboxes_actuales = [] # Lista de dicts: {"class_id": int, "x1": float, "y1": float, "x2": float, "y2": float}
        self.bbox_seleccionado = None
        self.clase_activa = tk.IntVar(value=0)
        self.mostrar_bboxes = tk.BooleanVar(value=True) 
        self.aislar_seleccionado = tk.BooleanVar(value=False) # <--- Nueva variable para aislar la vista
        
        # Variables de interacción del mouse
        self.estado_raton = "idle" # 'idle', 'creando', 'moviendo', 'redimensionando'
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        self.bbox_start_data = None

        self._construir_gui()

    # ==========================================
    # CONSTRUCCIÓN DE LA INTERFAZ
    # ==========================================

    def _construir_gui(self):
        # ---- Barra superior: seleccionar carpeta ----
        frame_top = tk.Frame(self.root, pady=8)
        frame_top.pack(fill="x")

        tk.Button(frame_top, text="Seleccionar carpeta de imágenes", command=self.seleccionar_carpeta).pack(side="left", padx=10)
        self.label_carpeta = tk.Label(frame_top, text="Ninguna carpeta seleccionada", fg="gray")
        self.label_carpeta.pack(side="left", padx=10)

        # ---- Contenedor central (Canvas + Panel Derecho) ----
        frame_central = tk.Frame(self.root)
        frame_central.pack(fill="both", expand=True, padx=10, pady=5)

        # Área del Canvas
        self.canvas_frame = tk.Frame(frame_central, bg="#222222")
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#222222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        
        # Eventos del Canvas
        self.canvas.bind("<Configure>", lambda e: self._dibujar_canvas())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Panel Derecho (Clases y Herramientas)
        panel_derecho = tk.Frame(frame_central, width=200, bd=2, relief="groove", padx=10, pady=10)
        panel_derecho.pack(side="right", fill="y", padx=(10, 0))

        # Checkbox Maestro para Mostrar/Ocultar BBoxes
        tk.Checkbutton(
            panel_derecho, 
            text="Mostrar Bounding Boxes", 
            variable=self.mostrar_bboxes, 
            command=self._dibujar_canvas,
            font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        # Checkbox Secundario para Aislar la vista
        tk.Checkbutton(
            panel_derecho, 
            text="Ocultar inactivos al editar", 
            variable=self.aislar_seleccionado, 
            command=self._dibujar_canvas,
            font=("Arial", 9)
        ).pack(anchor="w", padx=10, pady=(0, 15))

        tk.Label(panel_derecho, text="Clase Actual", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        for class_id, class_name in CLASES.items():
            rb = tk.Radiobutton(
                panel_derecho, 
                text=f"{class_id}: {class_name}", 
                variable=self.clase_activa, 
                value=class_id,
                command=self.al_cambiar_clase
            )
            rb.pack(anchor="w", pady=2)

        tk.Frame(panel_derecho, height=2, bd=1, relief="sunken").pack(fill="x", pady=15)
        
        # Botones de Acción para Bounding Boxes
        self.btn_borrar_box = tk.Button(panel_derecho, text="Borrar Box Seleccionado", command=self.borrar_box_seleccionado, bg="#f44336", fg="white", state="disabled")
        self.btn_borrar_box.pack(fill="x", pady=5)

        tk.Button(panel_derecho, text="Guardar Cambios", command=self.guardar_bboxes, bg="#4CAF50", fg="white").pack(fill="x", pady=5)

        # ---- Barra inferior de info ----
        frame_info = tk.Frame(self.root, pady=4)
        frame_info.pack(fill="x")

        self.label_contador_img = tk.Label(frame_info, text="Imagen: - / -", font=("Arial", 11, "bold"))
        self.label_contador_img.pack(side="left", padx=15)

        self.label_nombre_archivo = tk.Label(frame_info, text="", font=("Arial", 10))
        self.label_nombre_archivo.pack(side="left", padx=15)

        # ---- Barra inferior: navegación y borrado ----
        frame_botones = tk.Frame(self.root, pady=10)
        frame_botones.pack(fill="x")

        self.btn_anterior = tk.Button(frame_botones, text="< Anterior", width=15, command=self.imagen_anterior, state="disabled")
        self.btn_anterior.pack(side="left", padx=20)

        self.btn_borrar_img = tk.Button(frame_botones, text="Borrar Imagen Completa", width=22, bg="#d9534f", fg="white", activebackground="#c0392b", activeforeground="white", command=self.borrar_imagen_actual, state="disabled")
        self.btn_borrar_img.pack(side="left", expand=True)

        self.btn_siguiente = tk.Button(frame_botones, text="Siguiente >", width=15, command=self.imagen_siguiente, state="disabled")
        self.btn_siguiente.pack(side="right", padx=20)

        # ---- Atajos de teclado ----
        self.root.bind("<Right>", lambda e: self.imagen_siguiente())
        self.root.bind("<Left>", lambda e: self.imagen_anterior())
        self.root.bind("<Delete>", self.on_delete_key)
        # Añadido atajo rápido (tecla Escape) para deseleccionar el box actual
        self.root.bind("<Escape>", self.deseleccionar_box)

        tk.Label(self.root, text="Arrastrar = Crear/Mover | Clic = Seleccionar | Supr = Borrar Box | Esc = Deseleccionar", fg="gray", font=("Arial", 9)).pack(pady=(0, 6))

    # ==========================================
    # LÓGICA DE CARPETAS Y ARCHIVOS (FLEXIBLE)
    # ==========================================

    def seleccionar_carpeta(self):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta (puede contener images/labels o estar todo junto)")
        if not carpeta: return

        carpeta_images_test = os.path.join(carpeta, "images")
        carpeta_labels_test = os.path.join(carpeta, "labels")

        if os.path.isdir(carpeta_images_test) and os.path.isdir(carpeta_labels_test):
            self.carpeta_images = carpeta_images_test
            self.carpeta_labels = carpeta_labels_test
        else:
            self.carpeta_images = carpeta
            self.carpeta_labels = carpeta

        imagenes = sorted(f for f in os.listdir(self.carpeta_images) if f.lower().endswith(EXTENSIONES_IMAGEN))

        if not imagenes:
            messagebox.showwarning("Vacío", "No se encontraron imágenes en la carpeta seleccionada.")
            return

        self.carpeta_principal = carpeta
        self.lista_imagenes = imagenes
        self.indice_actual = 0
        
        self.label_carpeta.config(text=carpeta, fg="black")
        self.btn_anterior.config(state="normal")
        self.btn_siguiente.config(state="normal")
        self.btn_borrar_img.config(state="normal")

        self.cargar_imagen_actual()

    def cargar_imagen_actual(self):
        if not self.lista_imagenes: return

        nombre_imagen = self.lista_imagenes[self.indice_actual]
        imagen_path = os.path.join(self.carpeta_images, nombre_imagen)
        txt_path = self._ruta_txt_para(nombre_imagen)

        if not os.path.exists(imagen_path):
            del self.lista_imagenes[self.indice_actual]
            if self.indice_actual >= len(self.lista_imagenes):
                self.indice_actual = max(0, len(self.lista_imagenes) - 1)
            self.cargar_imagen_actual()
            return

        self.imagen_pil = Image.open(imagen_path).convert("RGB")
        self.img_w, self.img_h = self.imagen_pil.size
        
        self.cargar_bboxes_desde_txt(txt_path)
        self.bbox_seleccionado = None
        self._actualizar_estado_boton_borrar_box()
        
        self._dibujar_canvas()

        self.label_contador_img.config(text=f"Imagen: {self.indice_actual + 1} / {len(self.lista_imagenes)}")
        self.label_nombre_archivo.config(text=nombre_imagen)
        self.btn_anterior.config(state="normal" if self.indice_actual > 0 else "disabled")
        self.btn_siguiente.config(state="normal" if self.indice_actual < len(self.lista_imagenes) - 1 else "disabled")

    def _ruta_txt_para(self, nombre_imagen):
        nombre_base = os.path.splitext(nombre_imagen)[0]
        return os.path.join(self.carpeta_labels, nombre_base + ".txt")

    # ==========================================
    # LÓGICA DE BOUNDING BOXES (Unity Math)
    # ==========================================

    def cargar_bboxes_desde_txt(self, txt_path):
        self.bboxes_actuales = []
        if not os.path.exists(txt_path): return

        with open(txt_path, "r") as archivo:
            for linea in archivo:
                datos = linea.strip().split()
                if len(datos) < 5: continue
                try:
                    class_id = int(datos[0])
                    x_center, y_center, w, h = map(float, datos[1:5])
                    
                    x_center_px = x_center * self.img_w
                    y_center_px = ( y_center) * self.img_h
                    w_px = w * self.img_w
                    h_px = h * self.img_h

                    x1 = x_center_px - w_px / 2
                    y1 = y_center_px - h_px / 2
                    x2 = x_center_px + w_px / 2
                    y2 = y_center_px + h_px / 2

                    self.bboxes_actuales.append({
                        "class_id": class_id,
                        "x1": max(0, min(self.img_w, x1)),
                        "y1": max(0, min(self.img_h, y1)),
                        "x2": max(0, min(self.img_w, x2)),
                        "y2": max(0, min(self.img_h, y2))
                    })
                except ValueError:
                    continue

    def guardar_bboxes(self):
        if not self.lista_imagenes: return
        nombre_imagen = self.lista_imagenes[self.indice_actual]
        txt_path = self._ruta_txt_para(nombre_imagen)

        if not self.bboxes_actuales:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            return

        with open(txt_path, "w") as f:
            for box in self.bboxes_actuales:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                
                x_center_px = (x1 + x2) / 2.0
                y_center_px = (y1 + y2) / 2.0
                w_px = abs(x2 - x1)
                h_px = abs(y2 - y1)

                x_center = x_center_px / self.img_w
                y_center = (y_center_px / self.img_h)
                w = w_px / self.img_w
                h = h_px / self.img_h

                f.write(f"{box['class_id']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # ==========================================
    # RENDERIZADO DEL CANVAS
    # ==========================================

    def _dibujar_canvas(self):
        if not self.imagen_pil: return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10: return

        escala_x = canvas_w / self.img_w
        escala_y = canvas_h / self.img_h
        self.escala_actual = min(escala_x, escala_y)

        nuevo_w = int(self.img_w * self.escala_actual)
        nuevo_h = int(self.img_h * self.escala_actual)

        self.offset_x = (canvas_w - nuevo_w) // 2
        self.offset_y = (canvas_h - nuevo_h) // 2

        img_resized = self.imagen_pil.resize((max(1, nuevo_w), max(1, nuevo_h)), Image.LANCZOS)
        self.imagen_tk = ImageTk.PhotoImage(img_resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.imagen_tk, anchor="nw")

        # Dibujar rectángulos SOLO si la casilla principal está activada
        if self.mostrar_bboxes.get():
            for i, box in enumerate(self.bboxes_actuales):
                es_seleccionado = (i == self.bbox_seleccionado)
                
                # --- NUEVA LÓGICA DE AISLAMIENTO ---
                # Si está marcada la opción de aislar, y tenenos un box seleccionado en este momento,
                # ocultamos todos los que no sean el que estamos seleccionando/dibujando.
                if self.aislar_seleccionado.get() and self.bbox_seleccionado is not None:
                    if not es_seleccionado:
                        continue # Salta el dibujado de este box inactivo

                cx1, cy1 = self.img_to_canvas(box["x1"], box["y1"])
                cx2, cy2 = self.img_to_canvas(box["x2"], box["y2"])
                
                color = "#00FF00" if es_seleccionado else "red"
                grosor = 3 if es_seleccionado else 2

                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=grosor)
                nombre_clase = CLASES.get(box["class_id"], "Unknown")
                
                self.canvas.create_rectangle(cx1, cy1-20, cx1+(len(nombre_clase)*8)+20, cy1, fill=color, outline=color)
                self.canvas.create_text(cx1+5, cy1-10, text=f"{box['class_id']}:{nombre_clase}", fill="black", anchor="w", font=("Arial", 10, "bold"))

    # ==========================================
    # INTERACCIÓN DEL RATÓN Y TECLADO
    # ==========================================

    def img_to_canvas(self, ix, iy):
        return ix * self.escala_actual + self.offset_x, iy * self.escala_actual + self.offset_y

    def canvas_to_img(self, cx, cy):
        ix = (cx - self.offset_x) / self.escala_actual
        iy = (cy - self.offset_y) / self.escala_actual
        return max(0, min(self.img_w, ix)), max(0, min(self.img_h, iy))

    def on_mouse_press(self, event):
        if not self.mostrar_bboxes.get():
            self.mostrar_bboxes.set(True)

        ix, iy = self.canvas_to_img(event.x, event.y)
        
        caja_clic = None
        for i in range(len(self.bboxes_actuales)-1, -1, -1):
            box = self.bboxes_actuales[i]
            x_min, x_max = min(box["x1"], box["x2"]), max(box["x1"], box["x2"])
            y_min, y_max = min(box["y1"], box["y2"]), max(box["y1"], box["y2"])
            
            if x_min <= ix <= x_max and y_min <= iy <= y_max:
                caja_clic = i
                break

        if caja_clic is not None:
            self.bbox_seleccionado = caja_clic
            self.estado_raton = "moviendo"
            self.clase_activa.set(self.bboxes_actuales[caja_clic]["class_id"])
            self.mouse_start_x, self.mouse_start_y = ix, iy
            self.bbox_start_data = dict(self.bboxes_actuales[caja_clic])
        else:
            self.estado_raton = "creando"
            self.mouse_start_x, self.mouse_start_y = ix, iy
            
            nuevo_box = {
                "class_id": self.clase_activa.get(),
                "x1": ix, "y1": iy, "x2": ix, "y2": iy
            }
            self.bboxes_actuales.append(nuevo_box)
            self.bbox_seleccionado = len(self.bboxes_actuales) - 1

        self._actualizar_estado_boton_borrar_box()
        self._dibujar_canvas()

    def on_mouse_drag(self, event):
        if self.estado_raton == "idle" or self.bbox_seleccionado is None: return
        ix, iy = self.canvas_to_img(event.x, event.y)
        box = self.bboxes_actuales[self.bbox_seleccionado]

        if self.estado_raton == "creando":
            box["x2"], box["y2"] = ix, iy
            
        elif self.estado_raton == "moviendo":
            dx = ix - self.mouse_start_x
            dy = iy - self.mouse_start_y
            
            new_x1 = self.bbox_start_data["x1"] + dx
            new_y1 = self.bbox_start_data["y1"] + dy
            new_x2 = self.bbox_start_data["x2"] + dx
            new_y2 = self.bbox_start_data["y2"] + dy
            
            w = abs(new_x2 - new_x1)
            h = abs(new_y2 - new_y1)
            
            if new_x1 < 0: new_x1, new_x2 = 0, w
            if new_y1 < 0: new_y1, new_y2 = 0, h
            if new_x2 > self.img_w: new_x1, new_x2 = self.img_w - w, self.img_w
            if new_y2 > self.img_h: new_y1, new_y2 = self.img_h - h, self.img_h

            box["x1"], box["y1"], box["x2"], box["y2"] = new_x1, new_y1, new_x2, new_y2

        self._dibujar_canvas()

    def on_mouse_release(self, event):
        if self.estado_raton == "creando" and self.bbox_seleccionado is not None:
            box = self.bboxes_actuales[self.bbox_seleccionado]
            x_min, x_max = min(box["x1"], box["x2"]), max(box["x1"], box["x2"])
            y_min, y_max = min(box["y1"], box["y2"]), max(box["y1"], box["y2"])
            
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                self.bboxes_actuales.pop(self.bbox_seleccionado)
                self.bbox_seleccionado = None
            else:
                box["x1"], box["y1"], box["x2"], box["y2"] = x_min, y_min, x_max, y_max
                
        self.estado_raton = "idle"
        self._actualizar_estado_boton_borrar_box()
        self._dibujar_canvas()
        
    def deseleccionar_box(self, event=None):
        # Permite quitar la selección actual presionando la tecla Escape
        self.bbox_seleccionado = None
        self._actualizar_estado_boton_borrar_box()
        self._dibujar_canvas()

    def al_cambiar_clase(self):
        if self.bbox_seleccionado is not None:
            self.bboxes_actuales[self.bbox_seleccionado]["class_id"] = self.clase_activa.get()
            self._dibujar_canvas()

    # ==========================================
    # BORRADO (BOX E IMAGEN)
    # ==========================================

    def _actualizar_estado_boton_borrar_box(self):
        if self.bbox_seleccionado is not None:
            self.btn_borrar_box.config(state="normal")
        else:
            self.btn_borrar_box.config(state="disabled")

    def borrar_box_seleccionado(self):
        if self.bbox_seleccionado is not None:
            self.bboxes_actuales.pop(self.bbox_seleccionado)
            self.bbox_seleccionado = None
            self._actualizar_estado_boton_borrar_box()
            self._dibujar_canvas()

    def on_delete_key(self, event):
        if self.bbox_seleccionado is not None:
            self.borrar_box_seleccionado()
        else:
            self.borrar_imagen_actual()

    # ==========================================
    # NAVEGACIÓN Y BORRADO DE IMAGEN
    # ==========================================

    def imagen_siguiente(self):
        if self.lista_imagenes and self.indice_actual < len(self.lista_imagenes) - 1:
            self.guardar_bboxes() 
            self.indice_actual += 1
            self.cargar_imagen_actual()

    def imagen_anterior(self):
        if self.lista_imagenes and self.indice_actual > 0:
            self.guardar_bboxes() 
            self.indice_actual -= 1
            self.cargar_imagen_actual()

    def borrar_imagen_actual(self):
        if not self.lista_imagenes: return
        nombre_imagen = self.lista_imagenes[self.indice_actual]
        
        confirmar = messagebox.askyesno("Confirmar", f"¿Eliminar la imagen {nombre_imagen} y sus etiquetas de disco?")
        if not confirmar: return

        imagen_path = os.path.join(self.carpeta_images, nombre_imagen)
        txt_path = self._ruta_txt_para(nombre_imagen)

        try:
            if os.path.exists(imagen_path): os.remove(imagen_path)
            if os.path.exists(txt_path): os.remove(txt_path)
        except OSError as e:
            messagebox.showerror("Error", f"No se pudo borrar:\n{e}")
            return

        del self.lista_imagenes[self.indice_actual]
        if self.indice_actual >= len(self.lista_imagenes):
            self.indice_actual = max(0, len(self.lista_imagenes) - 1)

        if not self.lista_imagenes:
            self.canvas.delete("all")
            self.label_nombre_archivo.config(text="")
            messagebox.showinfo("Fin", "No hay más imágenes.")
        else:
            self.cargar_imagen_actual()

if __name__ == "__main__":
    root = tk.Tk()
    app = RevisorBoundingBoxes(root)
    root.mainloop()

# import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageDraw, ImageTk


# EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")


# class RevisorBoundingBoxes:
#     """
#     GUI para revisar bounding boxes (formato YOLO) dibujados sobre imágenes.

#     Estructura esperada de la carpeta principal:

#         carpeta_principal/
#             images/
#                 image_000230.png
#                 ...
#             labels/
#                 image_000230.txt
#                 ...

#     Al borrar una imagen se elimina PERMANENTEMENTE tanto el archivo de
#     imagen como su .txt de labels asociado. No hay papelera ni backup,
#     así que ¡ojo con el botón de borrar!
#     """

#     def __init__(self, root):
#         self.root = root
#         self.root.title("Revisor de Bounding Boxes - YOLO")
#         self.root.geometry("1100x780")
#         self.root.minsize(800, 600)

#         self.carpeta_principal = None
#         self.carpeta_images = None
#         self.carpeta_labels = None

#         self.lista_imagenes = []
#         self.indice_actual = 0
#         self.contador_borradas = 0

#         self.imagen_tk = None  # referencia viva para que Tk no la borre

#         self._construir_gui()

#     # ==========================================
#     # CONSTRUCCIÓN DE LA INTERFAZ
#     # ==========================================

#     def _construir_gui(self):

#         # ---- Barra superior: seleccionar carpeta ----
#         frame_top = tk.Frame(self.root, pady=8)
#         frame_top.pack(fill="x")

#         tk.Button(
#             frame_top,
#             text="Seleccionar carpeta principal",
#             command=self.seleccionar_carpeta,
#         ).pack(side="left", padx=10)

#         self.label_carpeta = tk.Label(
#             frame_top, text="Ninguna carpeta seleccionada", fg="gray"
#         )
#         self.label_carpeta.pack(side="left", padx=10)

#         # ---- Barra de información ----
#         frame_info = tk.Frame(self.root, pady=4)
#         frame_info.pack(fill="x")

#         self.label_contador_img = tk.Label(
#             frame_info, text="Imagen: - / -", font=("Arial", 11, "bold")
#         )
#         self.label_contador_img.pack(side="left", padx=15)

#         self.label_nombre_archivo = tk.Label(frame_info, text="", font=("Arial", 10))
#         self.label_nombre_archivo.pack(side="left", padx=15)

#         self.label_contador_borradas = tk.Label(
#             frame_info,
#             text="Imágenes borradas: 0",
#             font=("Arial", 11, "bold"),
#             fg="#c0392b",
#         )
#         self.label_contador_borradas.pack(side="right", padx=15)

#         # ---- Área de visualización de la imagen ----
#         self.canvas_frame = tk.Frame(self.root, bg="#222222")
#         self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

#         self.canvas = tk.Canvas(self.canvas_frame, bg="#222222", highlightthickness=0)
#         self.canvas.pack(fill="both", expand=True)
#         self.canvas.bind("<Configure>", lambda e: self._mostrar_imagen_actual())

#         # ---- Barra inferior: navegación y borrado ----
#         frame_botones = tk.Frame(self.root, pady=10)
#         frame_botones.pack(fill="x")

#         self.btn_anterior = tk.Button(
#             frame_botones,
#             text="< Anterior",
#             width=15,
#             command=self.imagen_anterior,
#             state="disabled",
#         )
#         self.btn_anterior.pack(side="left", padx=20)

#         self.btn_borrar = tk.Button(
#             frame_botones,
#             text="Borrar imagen",
#             width=22,
#             bg="#d9534f",
#             fg="white",
#             activebackground="#c0392b",
#             activeforeground="white",
#             command=self.borrar_imagen_actual,
#             state="disabled",
#         )
#         self.btn_borrar.pack(side="left", expand=True)

#         self.btn_siguiente = tk.Button(
#             frame_botones,
#             text="Siguiente >",
#             width=15,
#             command=self.imagen_siguiente,
#             state="disabled",
#         )
#         self.btn_siguiente.pack(side="right", padx=20)

#         # ---- Atajos de teclado ----
#         self.root.bind("<Right>", lambda e: self.imagen_siguiente())
#         self.root.bind("<Left>", lambda e: self.imagen_anterior())
#         self.root.bind("<Delete>", lambda e: self.borrar_imagen_actual())

#         tk.Label(
#             self.root,
#             text="Atajos: flecha derecha = siguiente | flecha izquierda = anterior | Supr = borrar",
#             fg="gray",
#             font=("Arial", 9),
#         ).pack(pady=(0, 6))

#     # ==========================================
#     # SELECCIÓN DE CARPETA
#     # ==========================================

#     def seleccionar_carpeta(self):
#         carpeta = filedialog.askdirectory(
#             title="Selecciona la carpeta principal (debe contener images/ y labels/)"
#         )
#         if not carpeta:
#             return

#         carpeta_images = os.path.join(carpeta, "images")
#         carpeta_labels = os.path.join(carpeta, "labels")

#         if not os.path.isdir(carpeta_images) or not os.path.isdir(carpeta_labels):
#             messagebox.showerror(
#                 "Estructura incorrecta",
#                 "La carpeta seleccionada debe contener las subcarpetas "
#                 "'images' y 'labels'.",
#             )
#             return

#         imagenes = sorted(
#             f
#             for f in os.listdir(carpeta_images)
#             if f.lower().endswith(EXTENSIONES_IMAGEN)
#         )

#         if not imagenes:
#             messagebox.showwarning(
#                 "Sin imágenes", "No se encontraron imágenes en la carpeta 'images'."
#             )
#             return

#         self.carpeta_principal = carpeta
#         self.carpeta_images = carpeta_images
#         self.carpeta_labels = carpeta_labels
#         self.lista_imagenes = imagenes
#         self.indice_actual = 0
#         self.contador_borradas = 0

#         self.label_contador_borradas.config(text="Imágenes borradas: 0")
#         self.label_carpeta.config(text=carpeta, fg="black")

#         self.btn_anterior.config(state="normal")
#         self.btn_siguiente.config(state="normal")
#         self.btn_borrar.config(state="normal")

#         self._mostrar_imagen_actual()

#     # ==========================================
#     # LÓGICA DE BOUNDING BOXES (adaptada del script original)
#     # ==========================================

#     def _ruta_txt_para(self, nombre_imagen):
#         nombre_base = os.path.splitext(nombre_imagen)[0]
#         return os.path.join(self.carpeta_labels, nombre_base + ".txt")

#     def _dibujar_bboxes_en_memoria(self, imagen_path, txt_path):
#         """Igual que dibujar_bounding_boxes() del script original, pero
#         devuelve la imagen en memoria en vez de guardarla en disco."""

#         imagen = Image.open(imagen_path).convert("RGB")
#         ancho, alto = imagen.size
#         draw = ImageDraw.Draw(imagen)

#         if not os.path.exists(txt_path):
#             return imagen

#         with open(txt_path, "r") as archivo:
#             lineas = archivo.readlines()

#         for linea in lineas:
#             linea = linea.strip()
#             if not linea:
#                 continue

#             datos = linea.split()
#             if len(datos) < 5:
#                 continue

#             try:
#                 class_id = int(datos[0])
#                 x_center = float(datos[1])
#                 y_center = float(datos[2])
#                 box_width = float(datos[3])
#                 box_height = float(datos[4])

#                 # Unity: y=0 abajo, y=1 arriba -> invertimos para PIL
#                 x_center_px = x_center * ancho
#                 y_center_px = (1.0 - y_center) * alto

#                 box_width_px = box_width * ancho
#                 box_height_px = box_height * alto

#                 x1 = int(x_center_px - box_width_px / 2)
#                 y1 = int(y_center_px - box_height_px / 2)
#                 x2 = int(x_center_px + box_width_px / 2)
#                 y2 = int(y_center_px + box_height_px / 2)

#                 x1 = max(0, min(ancho - 1, x1))
#                 y1 = max(0, min(alto - 1, y1))
#                 x2 = max(0, min(ancho - 1, x2))
#                 y2 = max(0, min(alto - 1, y2))

#                 draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
#                 draw.text((x1, max(0, y1 - 18)), f"Clase {class_id}", fill="red")

#             except ValueError:
#                 continue

#         return imagen

#     # ==========================================
#     # MOSTRAR IMAGEN ACTUAL
#     # ==========================================

#     def _mostrar_imagen_actual(self):
#         if not self.lista_imagenes:
#             return

#         nombre_imagen = self.lista_imagenes[self.indice_actual]
#         imagen_path = os.path.join(self.carpeta_images, nombre_imagen)
#         txt_path = self._ruta_txt_para(nombre_imagen)

#         if not os.path.exists(imagen_path):
#             # Pudo haber sido borrada por fuera del programa; refrescamos.
#             del self.lista_imagenes[self.indice_actual]
#             if self.indice_actual >= len(self.lista_imagenes):
#                 self.indice_actual = max(0, len(self.lista_imagenes) - 1)
#             self._mostrar_imagen_actual()
#             return

#         imagen_pil = self._dibujar_bboxes_en_memoria(imagen_path, txt_path)

#         canvas_w = self.canvas.winfo_width() or 900
#         canvas_h = self.canvas.winfo_height() or 550

#         img_w, img_h = imagen_pil.size
#         escala = min(canvas_w / img_w, canvas_h / img_h, 1.0)
#         nuevo_w = max(1, int(img_w * escala))
#         nuevo_h = max(1, int(img_h * escala))

#         imagen_mostrar = imagen_pil.resize((nuevo_w, nuevo_h), Image.LANCZOS)
#         self.imagen_tk = ImageTk.PhotoImage(imagen_mostrar)

#         self.canvas.delete("all")
#         self.canvas.create_image(
#             canvas_w // 2, canvas_h // 2, image=self.imagen_tk, anchor="center"
#         )

#         self.label_contador_img.config(
#             text=f"Imagen: {self.indice_actual + 1} / {len(self.lista_imagenes)}"
#         )
#         estado_txt = "con label" if os.path.exists(txt_path) else "SIN label"
#         self.label_nombre_archivo.config(text=f"{nombre_imagen}  ({estado_txt})")

#         self.btn_anterior.config(
#             state="normal" if self.indice_actual > 0 else "disabled"
#         )
#         self.btn_siguiente.config(
#             state="normal"
#             if self.indice_actual < len(self.lista_imagenes) - 1
#             else "disabled"
#         )

#     # ==========================================
#     # NAVEGACIÓN
#     # ==========================================

#     def imagen_siguiente(self):
#         if self.lista_imagenes and self.indice_actual < len(self.lista_imagenes) - 1:
#             self.indice_actual += 1
#             self._mostrar_imagen_actual()

#     def imagen_anterior(self):
#         if self.lista_imagenes and self.indice_actual > 0:
#             self.indice_actual -= 1
#             self._mostrar_imagen_actual()

#     # ==========================================
#     # BORRADO PERMANENTE (imagen + label)
#     # ==========================================

#     def borrar_imagen_actual(self):
#         if not self.lista_imagenes:
#             return

#         nombre_imagen = self.lista_imagenes[self.indice_actual]
#         imagen_path = os.path.join(self.carpeta_images, nombre_imagen)
#         txt_path = self._ruta_txt_para(nombre_imagen)

#         confirmar = messagebox.askyesno(
#             "Confirmar borrado",
#             f"¿Eliminar PERMANENTEMENTE:\n\n{nombre_imagen}\n"
#             f"y su archivo de etiquetas (.txt) asociado?\n\n"
#             f"Esta acción no se puede deshacer.",
#         )
#         if not confirmar:
#             return

#         try:
#             if os.path.exists(imagen_path):
#                 os.remove(imagen_path)
#             if os.path.exists(txt_path):
#                 os.remove(txt_path)
#         except OSError as e:
#             messagebox.showerror("Error al borrar", f"No se pudo borrar el archivo:\n{e}")
#             return

#         self.contador_borradas += 1
#         self.label_contador_borradas.config(
#             text=f"Imágenes borradas: {self.contador_borradas}"
#         )

#         del self.lista_imagenes[self.indice_actual]

#         if not self.lista_imagenes:
#             self.canvas.delete("all")
#             self.label_contador_img.config(text="Imagen: - / -")
#             self.label_nombre_archivo.config(text="")
#             self.btn_anterior.config(state="disabled")
#             self.btn_siguiente.config(state="disabled")
#             self.btn_borrar.config(state="disabled")
#             messagebox.showinfo("Listo", "No quedan más imágenes en la carpeta.")
#             return

#         if self.indice_actual >= len(self.lista_imagenes):
#             self.indice_actual = len(self.lista_imagenes) - 1

#         self._mostrar_imagen_actual()


# def main():
#     root = tk.Tk()
#     RevisorBoundingBoxes(root)
#     root.mainloop()


# if __name__ == "__main__":
#     main()