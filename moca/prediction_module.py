import os
import shutil
import threading
from tkinter import messagebox

import customtkinter as ctk
import joblib
import pandas as pd

from events import browse_file, restore_parent_window, browse_folder
from predict import exec_pred_task
from utils import load_json

pred_frame_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_pred_frame.json")
pred_frame_configs = load_json(pred_frame_configs_file_path)
widgets_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_widgets.json")
widgets_configs = load_json(widgets_configs_file_path)

lbl_text_style = widgets_configs["lbl_text_style"]
lbl_path_style = widgets_configs["lbl_path_style"]
btn_enb_state_style = widgets_configs["btn_enb_state_style"]
btn_dsb_state_style = widgets_configs["btn_dsb_state_style"]
btn_style = widgets_configs["btn_style"]

def open_prediction_window(parent_frame):
    parent_frame.withdraw()

    pred_root = ctk.CTkToplevel()
    pred_root.title(pred_frame_configs["pred_root_style"]["title"])
    pred_root.geometry(pred_frame_configs["pred_root_style"]["geometry"])
    pred_root.resizable(pred_frame_configs["pred_root_style"]["resizable"]["width"],
                        pred_frame_configs["pred_root_style"]["resizable"]["height"])
    pred_root.protocol(name="WM_DELETE_WINDOW",
                       func=lambda: restore_parent_window(child_root=pred_root, child_frame=pred_frame, parent_frame=parent_frame))

    pred_root.grid_rowconfigure(index=0, weight=1)
    pred_root.grid_columnconfigure(index=0, weight=1)

    pred_frame = ctk.CTkScrollableFrame(master=pred_root, **pred_frame_configs["pred_frame_args"])
    pred_frame.grid(row=0, column=0, sticky="nsew", **pred_frame_configs["pred_frame_grid"])

    pred_frame.grid_columnconfigure(index=0, weight=1)
    pred_frame.grid_rowconfigure(index=0, weight=1)

    # STEP 1: upload data file for prediction
    lbl_data_file_form = ctk.CTkLabel(master=pred_frame, **lbl_text_style, **pred_frame_configs["lbl_data_file_form_args"])
    lbl_data_file_form.grid(row=0, column=0, sticky="nsew", **pred_frame_configs["lbl_data_file_form_grid"])

    lbl_data_file_path = ctk.CTkLabel(master=pred_frame, **lbl_path_style, **pred_frame_configs["lbl_data_file_path_args"])
    lbl_data_file_path.grid(row=1, column=0, sticky="nsew", **pred_frame_configs["lbl_data_file_path_grid"])

    btn_browse_data_file = ctk.CTkButton(master=pred_frame, **btn_style, **pred_frame_configs["btn_browse_data_file_args"],
                                         command=lambda : browse_file(lbl_data_file_path))
    btn_browse_data_file.grid(row=2, column=0, sticky="nsew", **pred_frame_configs["btn_browse_data_file_grid"])

    # STEP 2: select reference dataset and model from internal folders
    lbl_refs_files_form = ctk.CTkLabel(master=pred_frame, **lbl_text_style, **pred_frame_configs["lbl_refs_files_form_args"])
    lbl_refs_files_form.grid(row=3, column=0, sticky="nsew", **pred_frame_configs["lbl_refs_files_form_grid"])

    lbl_refs_data_file_path = ctk.CTkLabel(master=pred_frame, **lbl_path_style, **pred_frame_configs["lbl_refs_data_file_path_args"])
    lbl_refs_data_file_path.grid(row=4, column=0, sticky="nsew", **pred_frame_configs["lbl_refs_data_file_path_grid"])

    lbl_refs_model_file_path = ctk.CTkLabel(master=pred_frame, **lbl_path_style, **pred_frame_configs["lbl_refs_model_file_path_args"])
    lbl_refs_model_file_path.grid(row=6, column=0, sticky="nsew", **pred_frame_configs["lbl_refs_model_file_path_grid"])

    btn_browse_refs_files = ctk.CTkButton(master=pred_frame, **btn_style, **pred_frame_configs["btn_browse_refs_files_args"],
                                          command=lambda : select_refs_files(lbl_refs_data_file_path, lbl_refs_model_file_path))
    btn_browse_refs_files.grid(row=7, column=0, sticky="nsew", **pred_frame_configs["btn_browse_refs_files_grid"])

    # STEP 3: select directory for output storing
    lbl_out_dir_form = ctk.CTkLabel(master=pred_frame, **lbl_text_style, **pred_frame_configs["lbl_out_dir_form_args"])
    lbl_out_dir_form.grid(row=8, column=0, sticky="nsew", **pred_frame_configs["lbl_out_dir_form_grid"])

    lbl_out_dir_path = ctk.CTkLabel(master=pred_frame, **lbl_path_style, **pred_frame_configs["lbl_out_dir_path_args"])
    lbl_out_dir_path.grid(row=9, column=0, sticky="nsew", **pred_frame_configs["lbl_out_dir_path_grid"])

    btn_browse_out_dir = ctk.CTkButton(master=pred_frame, **btn_style, **pred_frame_configs["btn_browse_out_dir_args"],
                                       command=lambda : browse_folder(lbl_out_dir_path))
    btn_browse_out_dir.grid(row=10, column=0, sticky="nsew", **pred_frame_configs["btn_browse_out_dir_grid"])

    # STEP 4: start prediction phase
    btn_run = ctk.CTkButton(master=pred_frame, **btn_enb_state_style, **pred_frame_configs["btn_run_args"],
                            command=lambda : start_prediction(lbl_data_file_path.cget("text"), lbl_refs_data_file_path.cget("text"),
                                                              lbl_refs_model_file_path.cget("text"), lbl_out_dir_path.cget("text"), btn_run))
    btn_run.grid(row=11, column=0, sticky="nsew", **pred_frame_configs["btn_run_grid"])


def start_prediction(data_path, refs_data_path, refs_model_path, out_dir, btn_run):
    err_msg_arr = ["No File Selected", "No Folder Selected", "No Dataset Selected", "No Model Selected"]
    if data_path in err_msg_arr:
        messagebox.showerror(title="Input Error", message="Please choose a valid data file for data prediction.")
    elif refs_model_path in err_msg_arr:
        messagebox.showerror(title="Input Error", message="Please choose a valid dataset and model for data prediction.")
    elif out_dir in err_msg_arr:
        messagebox.showerror(title="Input Error", message="Please choose a valid directory for storing results.")
    else:
        threading.Thread(target=exec_pred_task,
                         args=(data_path, refs_data_path, refs_model_path, out_dir, btn_run)).start()


def select_refs_files(refs_data_path, refs_model_path):
    data_path = ctk.filedialog.askopenfilename(initialdir="./datasets")

    if not data_path:
        refs_data_path.configure(text="No Dataset Selected")
        return

    model_list = []

    for filename in os.listdir("models"):
        model = os.path.join("models", filename)
        refs_model_pkg = joblib.load(model)
        refs_data = refs_model_pkg["data"]
        df1 = pd.read_excel(data_path)
        df2 = pd.read_excel(refs_data)

        if df1.equals(df2):
            model_list.append(model)

    tmp_dir = "Available Models"
    os.makedirs(tmp_dir, exist_ok=True)

    for file_path in model_list:
        shutil.copy(file_path, tmp_dir)

    model_path = ctk.filedialog.askopenfilename(initialdir=tmp_dir)

    if not model_path:
        refs_model_path.configure(text="No Model Selected")
        return

    return refs_data_path.configure(text=str(data_path)), refs_model_path.configure(text=str(model_path))

