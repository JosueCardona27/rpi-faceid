import tkinter as tk
from tkinter import messagebox
import hashlib
import sys
import os

# Asegurar que podemos importar database.py (está en el mismo directorio)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
# Importamos de tu database.py
try:
    from database import conectar, hash_contrasena
    print("[OK] database importado correctamente")
except ImportError as e:
    print(f"[ERROR] Importando database: {e}")
    # Fallback: definir conectar manualmente
    def conectar():
        import sqlite3
        db_path = os.path.join(current_dir, 'database', 'reconocimiento_facial.db')
        if not os.path.exists(db_path):
            project_root = os.path.dirname(current_dir)
            db_path = os.path.join(project_root, 'reconocimiento_facial.db')
        print(f"[DB] Conectando a: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Esto permite acceso por nombre
        return conn
    
    def hash_contrasena(password):
        return hashlib.sha256(password.encode()).hexdigest()

class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LabControl - Login")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        self.root.configure(bg="#0D0F14")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        frame = tk.Frame(self.root, bg="#0D0F14")
        frame.pack(expand=True, fill="both", padx=40, pady=40)
        
        # Título
        tk.Label(frame, text="LabControl", font=("Arial", 24, "bold"), 
                bg="#0D0F14", fg="#00D4FF").pack(pady=20)
        tk.Label(frame, text="Sistema de Gestión", font=("Arial", 12),
                bg="#0D0F14", fg="#6B7280").pack()
        
        # Campos
        tk.Label(frame, text="Usuario (Cuenta o Correo)", bg="#0D0F14", 
                fg="white", font=("Arial", 10)).pack(anchor="w", pady=(20,5))
        self.entry_user = tk.Entry(frame, font=("Arial", 12), width=30)
        self.entry_user.pack(fill="x", ipady=5)
        
        tk.Label(frame, text="Contraseña", bg="#0D0F14", 
                fg="white", font=("Arial", 10)).pack(anchor="w", pady=(15,5))
        self.entry_pass = tk.Entry(frame, font=("Arial", 12), width=30, show="*")
        self.entry_pass.pack(fill="x", ipady=5)
        
        # Botón
        btn = tk.Button(frame, text="Iniciar Sesión", command=self.login,
                       bg="#00D4FF", fg="#0D0F14", font=("Arial", 11, "bold"),
                       relief="flat", cursor="hand2")
        btn.pack(pady=30, fill="x", ipady=8)
        
        # Bind Enter
        self.entry_pass.bind('<Return>', lambda e: self.login())
        
    def login(self):
        usuario = self.entry_user.get().strip()
        password = self.entry_pass.get().strip()
        
        if not usuario or not password:
            messagebox.showerror("Error", "Complete todos los campos")
            return
            
        try:
            print(f"[LOGIN] Intentando conectar...")
            conn = conectar()
            cursor = conn.cursor()
            
            # Determinar si es correo o cuenta
            if '@' in usuario:
                cursor.execute("""
                    SELECT id, nombre, apellido_paterno, apellido_materno, 
                           numero_cuenta, correo, rol, contrasena 
                    FROM usuarios 
                    WHERE LOWER(correo) = LOWER(?) AND rol IN ('admin', 'maestro')
                """, (usuario,))
                print(f"[LOGIN] Buscando por correo: {usuario}")
            else:
                cursor.execute("""
                    SELECT id, nombre, apellido_paterno, apellido_materno,
                           numero_cuenta, correo, rol, contrasena 
                    FROM usuarios 
                    WHERE numero_cuenta = ? AND rol IN ('admin', 'maestro')
                """, (usuario,))
                print(f"[LOGIN] Buscando por cuenta: {usuario}")
            
            user = cursor.fetchone()
            conn.close()
            
            if not user:
                print("[LOGIN] Usuario no encontrado")
                messagebox.showerror("Error", "Usuario no encontrado o sin permisos")
                return
            
            # user es una tupla con el siguiente orden:
            # 0: id, 1: nombre, 2: apellido_paterno, 3: apellido_materno,
            # 4: numero_cuenta, 5: correo, 6: rol, 7: contrasena
            
            print(f"[LOGIN] Usuario encontrado: {user[1]} {user[2]} (ID: {user[0]})")
                
            # Verificar hash SHA-256
            hash_input = hashlib.sha256(password.encode()).hexdigest()
            print(f"[LOGIN] Hash calculado: {hash_input[:16]}...")
            
            # Acceder por índice (columna 7 es contrasena)
            contrasena_db = user[7]
            print(f"[LOGIN] Hash DB: {contrasena_db[:16]}...")
            
            if hash_input != contrasena_db:
                print("[LOGIN] Contraseña incorrecta")
                messagebox.showerror("Error", "Contraseña incorrecta")
                return
                
            print("[LOGIN] Autenticacion exitosa!")
            
            # Construir diccionario con los datos del usuario
            usuario_data = {
                'id': user[0],
                'nombre': user[1],
                'apellido_paterno': user[2],
                'apellido_materno': user[3],
                'numero_cuenta': user[4],
                'correo': user[5],
                'rol': user[6]
            }
            
            self.root.destroy()
            
            # Import dashboard aquí para no cargarlo al inicio
            try:
                # Agregar src/dashboard al path
                dashboard_path = os.path.join(current_dir, 'dashboard')
                if dashboard_path not in sys.path:
                    sys.path.insert(0, dashboard_path)
                
                # También agregar src/ al path para que dashboard pueda importar face_engine, etc.
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                
                from dashboard import Dashboard
                print("[LOGIN] Abriendo dashboard...")
                dash = Dashboard(usuario_data)
                dash.run()
            except Exception as e:
                print(f"[ERROR] No se pudo importar dashboard: {e}")
                import traceback
                error_completo = traceback.format_exc()
                print(error_completo)
                messagebox.showerror("Error", f"No se pudo abrir el dashboard:\n\n{str(e)}\n\nRevisa la consola para más detalles.")
            
        except Exception as e:
            print(f"[ERROR] Excepcion: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error de conexión: {str(e)}")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Crear usuario admin de prueba si no existe
    try:
        conn = conectar()
        cursor = conn.cursor()
        
        # Verificar si hay admin
        cursor.execute("SELECT COUNT(*) FROM usuarios WHERE rol = 'admin'")
        if cursor.fetchone()[0] == 0:
            print("[DB] Creando usuario admin por defecto...")
            # Crear admin por defecto
            cursor.execute("""
                INSERT INTO usuarios (nombre, apellido_paterno, apellido_materno,
                                     numero_cuenta, correo, contrasena, rol)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('Admin', 'Sistema', '', '20211813', 'admin@labcontrol.com',
                  hash_contrasena('admin123'), 'admin'))
            conn.commit()
            print("[DB] Usuario admin creado: 20211813 / admin123")
        else:
            print("[DB] Usuario admin ya existe")
        conn.close()
    except Exception as e:
        print(f"[DB] No se pudo verificar admin: {e}")
    
    print("[INIT] Iniciando aplicacion de login...")
    app = LoginWindow()
    app.run()