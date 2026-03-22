if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("[ERROR] Falta Pillow:  pip install Pillow")
        exit(1)
    
    from vistas.app_base import App

    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()