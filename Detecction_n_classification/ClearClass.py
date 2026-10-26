import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class DatasetCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Limpiador de Dataset GUI")
        self.root.geometry("900x750")
        self.root.configure(bg="#2e2e2e")

        self.folder_path = ""
        self.images = []
        self.current_index = 0
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

        self.setup_ui()
        self.bind_keys()

    def setup_ui(self):
        # Panel superior (Selección y contadores)
        top_frame = tk.Frame(self.root, bg="#2e2e2e")
        top_frame.pack(pady=10, fill=tk.X)

        self.btn_select = tk.Button(top_frame, text="Seleccionar Carpeta", font=("Arial", 11), command=self.select_folder)
        self.btn_select.pack(side=tk.LEFT, padx=20)

        # Etiqueta destacada para el progreso y total
        self.lbl_status = tk.Label(top_frame, text="0 / 0 | Totales: 0", font=("Arial", 12, "bold"), bg="#2e2e2e", fg="#4caf50")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        # Etiqueta para el nombre del archivo
        self.lbl_filename = tk.Label(top_frame, text="", font=("Arial", 11), bg="#2e2e2e", fg="white")
        self.lbl_filename.pack(side=tk.LEFT, padx=10)

        # Panel central (Imagen)
        self.lbl_image = tk.Label(self.root, bg="#1e1e1e", text="Carga una carpeta para comenzar", fg="gray", font=("Arial", 16))
        self.lbl_image.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Panel inferior (Controles)
        bottom_frame = tk.Frame(self.root, bg="#2e2e2e")
        bottom_frame.pack(pady=15)

        self.btn_prev = tk.Button(bottom_frame, text="<- Anterior [Izq]", font=("Arial", 12), bg="#555555", fg="white", width=15, command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.btn_delete = tk.Button(bottom_frame, text="Borrar [ d ]", font=("Arial", 12, "bold"), bg="#ff4c4c", fg="white", width=15, command=self.delete_image)
        self.btn_delete.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(bottom_frame, text="Siguiente -> [Der]", font=("Arial", 12), bg="#4caf50", fg="white", width=15, command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=10)

    def bind_keys(self):
        # Atajos de teclado
        self.root.bind("<space>", self.next_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("d", self.delete_image)
        self.root.bind("D", self.delete_image)
        self.root.bind("<Delete>", self.delete_image)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Selecciona la carpeta de la clase")
        if not folder:
            return

        self.folder_path = folder
        self.images = [f for f in os.listdir(folder) if f.lower().endswith(self.valid_extensions)]
        self.images.sort()
        
        if not self.images:
            messagebox.showinfo("Carpeta Vacía", "No se encontraron imágenes válidas.")
            return

        self.current_index = 0
        self.show_current_image()

    def show_current_image(self):
        total_images = len(self.images)
        
        if total_images == 0:
            self.lbl_image.configure(image='', text="¡Carpeta vaciada por completo!")
            self.lbl_image.image = None
            self.lbl_status.configure(text="0 / 0 | Totales: 0")
            self.lbl_filename.configure(text="")
            return

        # Pantalla final si te pasas de la última imagen
        if self.current_index >= total_images:
            self.lbl_image.configure(image='', text="¡Revisión terminada!\nHas llegado al final.\n\nUsa la flecha izquierda para retroceder.")
            self.lbl_image.image = None
            self.lbl_status.configure(text=f"Fin | Totales: {total_images}")
            self.lbl_filename.configure(text="")
            return

        img_name = self.images[self.current_index]
        img_path = os.path.join(self.folder_path, img_name)

        # Actualizar contadores
        self.lbl_status.configure(text=f"Imagen: {self.current_index + 1} / {total_images} | Total restantes: {total_images}")
        self.lbl_filename.configure(text=f"Archivo: {img_name}")

        try:
            img = Image.open(img_path)
            img.thumbnail((800, 550), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.lbl_image.configure(image=self.photo, text="")
        except Exception as e:
            print(f"Error al cargar {img_name}: {e}")
            self.lbl_image.configure(image='', text=f"Error al cargar la imagen.\n{e}")

    def next_image(self, event=None):
        # Permite avanzar hasta llegar a la pantalla final
        if self.images and self.current_index < len(self.images):
            self.current_index += 1
            self.show_current_image()

    def prev_image(self, event=None):
        # Retroceder sin pasar del índice 0
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

    def delete_image(self, event=None):
        # Bloquear borrado si estamos en la pantalla final o no hay imágenes
        if not self.images or self.current_index >= len(self.images):
            return

        img_name = self.images[self.current_index]
        img_path = os.path.join(self.folder_path, img_name)

        try:
            os.remove(img_path)
            print(f"Eliminada: {img_name}")
            del self.images[self.current_index]
            
            # Nota: No sumamos ni restamos al current_index. 
            # Al borrar el elemento en la posición actual, la siguiente imagen 
            # automáticamente cae en este mismo índice.
            
            self.show_current_image()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo borrar {img_name}.\nError: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCleanerApp(root)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()