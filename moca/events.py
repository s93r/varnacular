from pathlib import Path

import customtkinter as ctk

from tools import get_xls_file_format_list, get_txt_file_format_list


def restore_parent_window(child_root, child_frame, parent_frame):
    child_root.destroy()
    child_frame.destroy()
    parent_frame.deiconify()


def browse_file(lbl_dataset_filepath):
    file_path = ctk.filedialog.askopenfilename()

    ext_list = get_xls_file_format_list() + get_txt_file_format_list()
    ext = Path(file_path).suffix.lstrip('.').lower()

    if ext in ext_list:
        lbl_dataset_filepath.configure(text=str(file_path))
    else:
        lbl_dataset_filepath.configure(text="No File Selected")


def browse_folder(lbl_folder_path):
    folder_selected = ctk.filedialog.askdirectory()

    if folder_selected:
        lbl_folder_path.configure(text=str(folder_selected))
    else:
        lbl_folder_path.configure(text="No Folder Selected")




