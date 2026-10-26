# archivos = ['VehicleClassificationDataset/analysis/extreme_aspect_ratios.csv', 'VehicleClassificationDataset/analysis/extreme_dimensions.csv']

import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import re

EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")

CLASES = {
    0: "Car",
    1: "Motorcycle",
    2: "Passenger",
    3: "Cargo",
    4: "Ambulance",
    5: "Police",
    6: "FireTruck"
}

def parsear_rutas_y_agrupar():
    archivos = ['VehicleClassificationDataset/analysis/extreme_aspect_ratios.csv', 'VehicleClassificationDataset/analysis/extreme_dimensions.csv']

    rutas_crops = set()
    for f in archivos:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                if 'path' in df.columns:
                    rutas_crops.update(df['path'].dropna().tolist())
            except Exception as e:
                print(f"Error leyendo {f}: {e}")
    
    agrupaciones = {}
    for crop_path in rutas_crops:
        nombre_crop = os.path.basename(crop_path)
        match = re.match(r"(YOLODataset.*?)_(image_\d+)_(\d+)\.(jpg|png|jpeg)", nombre_crop)
        
        if match:
            dataset_folder = match.group(1)
            image_name_base = match.group(2)
            
            dir_clase = os.path.dirname(crop_path) 
            dir_vehicle_class_dataset = os.path.dirname(dir_clase) 
            dir_proyecto = os.path.dirname(dir_vehicle_class_dataset) 
            
            ruta_intento_png = os.path.join(dir_proyecto, dataset_folder, "images", f"{image_name_base}.png")
            ruta_intento_jpg = os.path.join(dir_proyecto, dataset_folder, "images", f"{image_name_base}.jpg")
            
            ruta_original = ruta_intento_png
            if os.path.exists(ruta_intento_jpg):
                ruta_original = ruta_intento_jpg
                
            if ruta_original not in agrupaciones:
                agrupaciones[ruta_original] = []
            agrupaciones[ruta_original].append(crop_path)
        else:
             agrupaciones[crop_path] = [crop_path]
             
    resultado = [{"original": orig, "crops": sorted(crops)} for orig, crops in agrupaciones.items()]
    return sorted(resultado, key=lambda x: x["original"])

class RevisorAgrupadoInteractivo:
    def __init__(self, root):
        self.root = root
        self.root.title("Anotador y Revisor de Casos Extremos (Original + Crops)")
        self.root.geometry("1500x850") 
        self.root.minsize(1000, 600)
        
        self.datos_agrupados = parsear_rutas_y_agrupar()
        self.indice_actual = 0
        
        self.img_tk_original = None
        self.imagen_pil = None
        self.imgs_tk_crops = []
        
        # Variables Render y Bounding Boxes
        self.img_w = 0
        self.img_h = 0
        self.escala_actual = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.bboxes_actuales = []
        self.indices_sospechosos = []
        
        self.bbox_seleccionado = None
        self.clase_activa = tk.IntVar(value=0)
        self.mostrar_bboxes = tk.BooleanVar(value=True) 
        self.aislar_seleccionado = tk.BooleanVar(value=False)
        
        # Interacción Ratón
        self.estado_raton = "idle" 
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        self.bbox_start_data = None
        
        self._construir_gui()
        
        if not self.datos_agrupados:
            messagebox.showerror("Error", "No se encontraron datos en los CSVs.")
        else:
            self.cargar_grupo_actual()
            
    def _construir_gui(self):
        # ---- Top Frame ----
        frame_top = tk.Frame(self.root, pady=10)
        frame_top.pack(fill="x")
        self.lbl_info = tk.Label(frame_top, text="", font=("Arial", 12, "bold"))
        self.lbl_info.pack()
        self.lbl_path = tk.Label(frame_top, text="", font=("Arial", 9), fg="gray")
        self.lbl_path.pack()

        # ---- Center Frame (PanedWindow) ----
        self.frame_central = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.frame_central.pack(fill="both", expand=True, padx=10, pady=5)
        
        # === Panel Izquierdo (Herramientas + Canvas Original) ===
        self.panel_izq = tk.Frame(self.frame_central, bg="#333", width=900)
        
        # Herramientas de Edición (Mini Panel Izquierdo)
        panel_herr = tk.Frame(self.panel_izq, width=180, bd=2, relief="groove", padx=5, pady=5)
        panel_herr.pack(side="left", fill="y")
        
        tk.Label(panel_herr, text="Editor Original", font=("Arial", 11, "bold")).pack(pady=(5,10))
        
        tk.Checkbutton(panel_herr, text="Mostrar BBoxes", variable=self.mostrar_bboxes, command=self._dibujar_canvas, font=("Arial", 9, "bold")).pack(anchor="w", pady=2)
        tk.Checkbutton(panel_herr, text="Ocultar inactivos", variable=self.aislar_seleccionado, command=self._dibujar_canvas, font=("Arial", 9)).pack(anchor="w", pady=(0, 10))
        
        tk.Label(panel_herr, text="Clase Activa:", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        for class_id, class_name in CLASES.items():
            tk.Radiobutton(panel_herr, text=f"{class_id}: {class_name}", variable=self.clase_activa, value=class_id, command=self.al_cambiar_clase).pack(anchor="w")

        tk.Frame(panel_herr, height=2, bd=1, relief="sunken").pack(fill="x", pady=15)
        
        self.btn_borrar_box = tk.Button(panel_herr, text="Borrar Box\nSeleccionado", command=self.borrar_box_seleccionado, bg="#f44336", fg="white", state="disabled")
        self.btn_borrar_box.pack(fill="x", pady=5)
        
        tk.Button(panel_herr, text="Guardar Cambios\nen Original .txt", command=self.guardar_bboxes, bg="#4CAF50", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=15)

        # Canvas Original
        self.canvas_orig = tk.Canvas(self.panel_izq, bg="#222", cursor="crosshair", highlightthickness=0)
        self.canvas_orig.pack(side="left", fill="both", expand=True)
        self.canvas_orig.bind("<Configure>", lambda e: self._dibujar_canvas())
        self.canvas_orig.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas_orig.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.frame_central.add(self.panel_izq)
        
        # === Panel Derecho (Crops de Referencia) ===
        self.panel_der = tk.Frame(self.frame_central, width=500)
        
        self.canvas_crops = tk.Canvas(self.panel_der)
        self.scrollbar = tk.Scrollbar(self.panel_der, orient="vertical", command=self.canvas_crops.yview)
        self.scrollable_frame = tk.Frame(self.canvas_crops)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_crops.configure(scrollregion=self.canvas_crops.bbox("all")))
        self.canvas_crops.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_crops.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas_crops.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.frame_central.add(self.panel_der)
        
        # ---- Bottom Frame (Navegación) ----
        frame_botones = tk.Frame(self.root, pady=10)
        frame_botones.pack(fill="x")
        
        self.btn_ant = tk.Button(frame_botones, text="< Grupo Anterior", width=15, command=self.anterior)
        self.btn_ant.pack(side="left", padx=20)
        
        self.btn_borrar_todo = tk.Button(frame_botones, text="Borrar TODO (Original + Todos sus Crops y TXTs)", bg="#d9534f", fg="white", font=("Arial", 10, "bold"), command=self.borrar_grupo_completo)
        self.btn_borrar_todo.pack(side="left", expand=True)
        
        self.btn_sig = tk.Button(frame_botones, text="Siguiente Grupo >", width=15, command=self.siguiente)
        self.btn_sig.pack(side="right", padx=20)
        
        # Atajos
        self.root.bind("<Right>", lambda e: self.siguiente())
        self.root.bind("<Left>", lambda e: self.anterior())
        self.root.bind("<Delete>", self.on_delete_key)
        self.root.bind("<Escape>", self.deseleccionar_box)

    # ==========================================
    # LOGICA DE DATOS Y RENDERIZADO
    # ==========================================

    def cargar_grupo_actual(self):
        if not self.datos_agrupados:
            self.canvas_orig.delete("all")
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            return
            
        grupo = self.datos_agrupados[self.indice_actual]
        ruta_orig = grupo["original"]
        rutas_crops = grupo["crops"]
        
        self.lbl_info.config(text=f"Grupo {self.indice_actual + 1} / {len(self.datos_agrupados)} | Crops problemáticos: {len(rutas_crops)}")
        self.lbl_path.config(text=ruta_orig)
        
        # Procesar los índices sospechosos desde los nombres de los crops
        self.indices_sospechosos = []
        for cp in rutas_crops:
            match = re.search(r'_(\d+)\.(jpg|png|jpeg)$', os.path.basename(cp), re.IGNORECASE)
            if match:
                self.indices_sospechosos.append(int(match.group(1)))
                
        # Cargar Imagen Original
        self.bbox_seleccionado = None
        self._actualizar_estado_boton_borrar_box()
        
        if os.path.exists(ruta_orig):
            self.imagen_pil = Image.open(ruta_orig).convert("RGB")
            self.img_w, self.img_h = self.imagen_pil.size
            self.cargar_bboxes_desde_txt(self.obtener_ruta_txt(ruta_orig))
            self._dibujar_canvas()
        else:
            self.imagen_pil = None
            self.canvas_orig.delete("all")
            self.canvas_orig.create_text(400, 400, text="IMAGEN ORIGINAL NO ENCONTRADA\n\n(Ruta buscada:\n" + ruta_orig + ")", fill="red", font=("Arial", 14, "bold"), justify="center")
            
        # Cargar Crops Referencia
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.imgs_tk_crops = []
        for i, crop_path in enumerate(rutas_crops):
            frame_crop = tk.Frame(self.scrollable_frame, pady=10, bd=1, relief="solid")
            frame_crop.pack(fill="x", padx=10, pady=10)
            
            idx_crop = self.indices_sospechosos[i] if i < len(self.indices_sospechosos) else "?"
            lbl_name = tk.Label(frame_crop, text=f"Índice BBox: {idx_crop} | {os.path.basename(crop_path)}", font=("Arial", 10, "bold"), fg="#d35400")
            lbl_name.pack()
            
            if os.path.exists(crop_path):
                c_img = Image.open(crop_path)
                w, h = c_img.size
                c_img.thumbnail((450, 450)) # Crop grande
                tk_img = ImageTk.PhotoImage(c_img)
                self.imgs_tk_crops.append(tk_img)
                tk.Label(frame_crop, image=tk_img).pack()
                
                info = f"W:{w} H:{h} | AR:{w/h:.2f}"
                tk.Label(frame_crop, text=info, fg="red", font=("Arial", 10, "bold")).pack(pady=3)
                
                tk.Button(frame_crop, text="Borrar Archivo de este Crop (.jpg y .txt)", bg="#f0ad4e", command=lambda cp=crop_path, idx=i: self.borrar_crop_individual(cp, idx)).pack(pady=5)
            else:
                tk.Label(frame_crop, text="Crop no encontrado en disco", fg="red").pack()

    def obtener_ruta_txt(self, img_path):
        dir_base = os.path.dirname(img_path)
        nombre_base = os.path.splitext(os.path.basename(img_path))[0]
        if os.path.basename(dir_base) == "images":
            return os.path.join(os.path.dirname(dir_base), "labels", nombre_base + ".txt")
        return os.path.join(dir_base, nombre_base + ".txt")

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
                    y_center_px = (1.0 - y_center) * self.img_h
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
        if not self.datos_agrupados or not self.imagen_pil: return
        ruta_orig = self.datos_agrupados[self.indice_actual]["original"]
        txt_path = self.obtener_ruta_txt(ruta_orig)

        if not self.bboxes_actuales:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            messagebox.showinfo("Guardado", "Se borró el archivo .txt porque no quedan bounding boxes.")
            return

        with open(txt_path, "w") as f:
            for box in self.bboxes_actuales:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                x_center_px = (x1 + x2) / 2.0
                y_center_px = (y1 + y2) / 2.0
                w_px = abs(x2 - x1)
                h_px = abs(y2 - y1)

                x_center = x_center_px / self.img_w
                y_center = 1.0 - (y_center_px / self.img_h) # Unity
                w = w_px / self.img_w
                h = h_px / self.img_h

                f.write(f"{box['class_id']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        messagebox.showinfo("Guardado", "Bounding boxes guardados en el archivo original.")

    def _dibujar_canvas(self):
        if not self.imagen_pil: return

        canvas_w = self.canvas_orig.winfo_width()
        canvas_h = self.canvas_orig.winfo_height()
        if canvas_w < 10 or canvas_h < 10: return

        escala_x = canvas_w / self.img_w
        escala_y = canvas_h / self.img_h
        self.escala_actual = min(escala_x, escala_y)

        nuevo_w = int(self.img_w * self.escala_actual)
        nuevo_h = int(self.img_h * self.escala_actual)

        self.offset_x = (canvas_w - nuevo_w) // 2
        self.offset_y = (canvas_h - nuevo_h) // 2

        img_resized = self.imagen_pil.resize((max(1, nuevo_w), max(1, nuevo_h)), Image.LANCZOS)
        self.img_tk_original = ImageTk.PhotoImage(img_resized)
        
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(self.offset_x, self.offset_y, image=self.img_tk_original, anchor="nw")

        if self.mostrar_bboxes.get():
            for i, box in enumerate(self.bboxes_actuales):
                es_seleccionado = (i == self.bbox_seleccionado)
                es_sospechoso = (i in self.indices_sospechosos)
                
                if self.aislar_seleccionado.get() and self.bbox_seleccionado is not None:
                    if not es_seleccionado:
                        continue 

                cx1, cy1 = self.img_to_canvas(box["x1"], box["y1"])
                cx2, cy2 = self.img_to_canvas(box["x2"], box["y2"])
                
                # Sistema de Colores: Verde=Seleccionado, Amarillo=Extremo/Sospechoso, Rojo=Normal
                if es_seleccionado:
                    color = "#00FF00"
                    grosor = 3
                elif es_sospechoso:
                    color = "yellow"
                    grosor = 4
                else:
                    color = "red"
                    grosor = 2

                self.canvas_orig.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=grosor)
                nombre_clase = CLASES.get(box["class_id"], "Unknown")
                
                if es_sospechoso and not es_seleccionado:
                    texto_label = f"[{i}] {nombre_clase} (!)"
                else:
                    texto_label = f"[{i}] {nombre_clase}"
                    
                self.canvas_orig.create_rectangle(cx1, cy1-20, cx1+(len(texto_label)*8)+10, cy1, fill=color, outline=color)
                self.canvas_orig.create_text(cx1+5, cy1-10, text=texto_label, fill="black", anchor="w", font=("Arial", 10, "bold"))

    # ==========================================
    # INTERACCION RATON (Igual al Script 1)
    # ==========================================

    def img_to_canvas(self, ix, iy):
        return ix * self.escala_actual + self.offset_x, iy * self.escala_actual + self.offset_y

    def canvas_to_img(self, cx, cy):
        ix = (cx - self.offset_x) / self.escala_actual
        iy = (cy - self.offset_y) / self.escala_actual
        return max(0, min(self.img_w, ix)), max(0, min(self.img_h, iy))

    def on_mouse_press(self, event):
        if not self.mostrar_bboxes.get(): self.mostrar_bboxes.set(True)
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
            nuevo_box = {"class_id": self.clase_activa.get(), "x1": ix, "y1": iy, "x2": ix, "y2": iy}
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
            
            w, h = abs(new_x2 - new_x1), abs(new_y2 - new_y1)
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
        self.bbox_seleccionado = None
        self._actualizar_estado_boton_borrar_box()
        self._dibujar_canvas()

    def al_cambiar_clase(self):
        if self.bbox_seleccionado is not None:
            self.bboxes_actuales[self.bbox_seleccionado]["class_id"] = self.clase_activa.get()
            self._dibujar_canvas()

    # ==========================================
    # NAVEGACION Y BORRADO
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

    def siguiente(self):
        if self.indice_actual < len(self.datos_agrupados) - 1:
            self.indice_actual += 1
            self.cargar_grupo_actual()

    def anterior(self):
        if self.indice_actual > 0:
            self.indice_actual -= 1
            self.cargar_grupo_actual()
            
    def borrar_crop_individual(self, crop_path, index_in_ui):
        resp = messagebox.askyesno("Borrar Archivo Crop", "¿Borrar el archivo .jpg y .txt de este crop del disco?\n(Nota: Esto no borra la coordenada de la imagen original)")
        if resp:
            txt_path = self.obtener_ruta_txt(crop_path)
            if os.path.exists(crop_path): os.remove(crop_path)
            if os.path.exists(txt_path): os.remove(txt_path)
            
            self.datos_agrupados[self.indice_actual]["crops"].remove(crop_path)
            if len(self.datos_agrupados[self.indice_actual]["crops"]) == 0:
                self.datos_agrupados.pop(self.indice_actual)
                if self.indice_actual >= len(self.datos_agrupados):
                    self.indice_actual -= 1
            self.cargar_grupo_actual()

    def borrar_grupo_completo(self):
        grupo = self.datos_agrupados[self.indice_actual]
        resp = messagebox.askyesno("PELIGRO", f"¿Borrar la imagen original Y TODOS LOS CROPS ({len(grupo['crops'])}) de esta vista? (Se borrarán imágenes y .txt del disco)")
        if resp:
            orig_img = grupo["original"]
            orig_txt = self.obtener_ruta_txt(orig_img)
            if os.path.exists(orig_img): os.remove(orig_img)
            if os.path.exists(orig_txt): os.remove(orig_txt)
            
            for cp in grupo["crops"]:
                cp_txt = self.obtener_ruta_txt(cp)
                if os.path.exists(cp): os.remove(cp)
                if os.path.exists(cp_txt): os.remove(cp_txt)
                
            self.datos_agrupados.pop(self.indice_actual)
            if self.indice_actual >= len(self.datos_agrupados):
                self.indice_actual = max(0, len(self.datos_agrupados) - 1)
            self.cargar_grupo_actual()

if __name__ == '__main__':
    root = tk.Tk()
    app = RevisorAgrupadoInteractivo(root)
    root.mainloop()