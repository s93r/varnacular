import customtkinter as ctk

def open_models_catalog_window():
    ctk.filedialog.askopenfilename(initialdir="./models")

