import customtkinter as ctk

def open_datasets_catalog_window():
    ctk.filedialog.askopenfilename(initialdir="./datasets")