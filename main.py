import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import sqlite3
import numpy as np


class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition App")
        self.window.minsize(800, 600)  # Минимальный размер окна
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=0)



        # Инициализация GUI
        self.video_label = tk.Label(window)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_panel = tk.Frame(window)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.name_entry = tk.Entry(self.control_panel, width=20)
        self.name_entry.pack(pady=5)

        self.button_frame = tk.Frame(self.control_panel)
        self.button_frame.pack(pady=5)

        self.add_button = tk.Button(self.button_frame,
                                    text="ADD TO DB",
                                    command=self.add_to_db,
                                    state=tk.DISABLED,
                                    height=3)
        self.add_button.pack(side=tk.LEFT, padx=2)

        self.delete_button = tk.Button(self.button_frame,
                                       text="DELETE FROM DB",
                                       command=self.delete_from_db,
                                       state=tk.DISABLED,
                                       height=3)
        self.delete_button.pack(side=tk.LEFT, padx=2)


        self.db_list = tk.Listbox(self.control_panel, width=30)
        self.db_list.pack(pady=5, fill=tk.BOTH, expand=True)
        self.db_list.bind('<<ListboxSelect>>', self.on_list_select)

        # Инициализация базы данных
        self.conn = sqlite3.connect('faces.db')
        self.create_db_table()
        self.load_db_data()

        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.current_frame = None
        self.is_adding = False

        # Запуск обновления видео
        self.update_video()

        # Обработка закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_list_select(self, event):
        if self.db_list.curselection():
            self.delete_button.config(state=tk.NORMAL)
        else:
            self.delete_button.config(state=tk.DISABLED)

    def delete_from_db(self):
        selected = self.db_list.curselection()
        if not selected:
            messagebox.showerror("ERROR", "SELECT ITEM FOR DELETE")
            return

        selected_name = self.db_list.get(selected[0])

        confirm = messagebox.askyesno(
            "ARE YOU SURE?",
            f"DO YOU WANT DELETE '{selected_name}'?"
        )
        if not confirm:
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM faces WHERE name = ?", (selected_name,))
            self.conn.commit()
            self.load_db_data()
            messagebox.showinfo("SUCCESS", "DELETED")
        except Exception as e:
            messagebox.showerror("ERROR", f"ERROR: {str(e)}")

        self.delete_button.config(state=tk.DISABLED)
    def create_db_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT NOT NULL,
                           embedding BLOB NOT NULL)''')
        self.conn.commit()

    def load_db_data(self):
        self.db_list.delete(0, tk.END)
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM faces")
        for row in cursor.fetchall():
            self.db_list.insert(tk.END, row[0])

    def update_video(self):
        if not self.is_adding:
            ret, frame = self.cap.read()
            if ret:
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()

                #label_width = self.window

                # Масштабируем кадр под размеры окна
                #if label_width > 0 and label_height > 0:
                #frame = cv2.resize(frame, (1200, 800))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = rgb_frame.copy()

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    height = bottom - top
                    color = (0, 255, 0)
                    label = ""

                    if height > 200:
                        color = (0, 0, 255)
                        self.add_button.config(state=tk.NORMAL)
                    else:
                        self.add_button.config(state=tk.DISABLED)

                    matches = self.check_face_in_db(face_encoding)
                    if not matches:
                        #label = "Нет в базе"
                        label = "UNKNOWN"
                    else:
                        label = f"HELLO {matches[0]}"

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def check_face_in_db(self, face_encoding):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, embedding FROM faces")
        for name, embedding_bytes in cursor.fetchall():
            stored_encoding = np.frombuffer(embedding_bytes, dtype=np.float64)
            match = face_recognition.compare_faces([stored_encoding], face_encoding)
            if match[0]:
                return [name]
        return []

    def add_to_db(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Ошибка", "Введите имя")
            return

        self.is_adding = True
        self.add_button.config(state=tk.DISABLED)

        face_locations = face_recognition.face_locations(self.current_frame)
        if not face_locations:
            messagebox.showerror("ERROR", "NO FACE IN DB")
            self.is_adding = False
            return

        top, right, bottom, left = face_locations[0]
        face_encoding = face_recognition.face_encodings(self.current_frame, [(top, right, bottom, left)])[0]

        embedding_bytes = face_encoding.tobytes()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",
                       (name, embedding_bytes))
        self.conn.commit()

        self.load_db_data()
        self.is_adding = False
        self.name_entry.delete(0, tk.END)
        messagebox.showinfo("SUCCESS", "FACE WAS ADDED TO DB")

    def on_closing(self):
        self.cap.release()
        self.conn.close()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()