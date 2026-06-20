import os
import threading
from tkinter import messagebox, ttk

import customtkinter as ctk

from events import browse_file, browse_folder, restore_parent_window
from tools import get_train_model_list, get_fsel_model_list, get_cv_dict
from train import exec_train_task
from utils import load_json

train_frame_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_train_frame.json")
train_frame_configs = load_json(train_frame_configs_file_path)
widgets_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_widgets.json")
widgets_configs = load_json(widgets_configs_file_path)

lbl_head_style = widgets_configs["lbl_head_style"]
lbl_text_style = widgets_configs["lbl_text_style"]
lbl_path_style = widgets_configs["lbl_path_style"]
btn_enb_state_style = widgets_configs["btn_enb_state_style"]
btn_dsb_state_style = widgets_configs["btn_dsb_state_style"]
menu_style = widgets_configs["menu_style"]
btn_style = widgets_configs["btn_style"]

cv_list = list(get_cv_dict().keys())
fsel_model_list = get_fsel_model_list()
train_model_list = get_train_model_list()

def open_training_window(parent_frame):
    parent_frame.withdraw()

    train_root = ctk.CTkToplevel()
    train_root.title(train_frame_configs["train_root_style"]["title"])
    train_root.geometry(train_frame_configs["train_root_style"]["geometry"])
    train_root.resizable(train_frame_configs["train_root_style"]["resizable"]["width"],
                         train_frame_configs["train_root_style"]["resizable"]["height"])
    train_root.protocol(name="WM_DELETE_WINDOW",
                        func=lambda : restore_parent_window(child_root=train_root, child_frame=train_frame, parent_frame=parent_frame))

    train_root.grid_rowconfigure(index=0, weight=1)
    train_root.grid_columnconfigure(index=0, weight=1)

    train_frame = ctk.CTkScrollableFrame(master=train_root, **train_frame_configs["train_frame_args"])
    train_frame.grid(row=0, column=0, sticky="nsew", **train_frame_configs["train_frame_grid"])

    train_frame.grid_columnconfigure(index=0, weight=1)
    train_frame.grid_rowconfigure(index=0, weight=1)

    # HEADING 1: upload data
    lbl_data_file_head = ctk.CTkLabel(master=train_frame, **lbl_head_style, **train_frame_configs["lbl_data_file_head_args"], font=("Roboto", 13, "bold"))
    lbl_data_file_head.grid(row=0, column=0, sticky="nsew", **train_frame_configs["lbl_data_file_head_grid"])

    # STEP 1: upload data file for model training
    lbl_data_file_text = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_data_file_text_args"])
    lbl_data_file_text.grid(row=1, column=0, sticky="nsew", **train_frame_configs["lbl_data_file_text_grid"])

    lbl_data_file_path = ctk.CTkLabel(master=train_frame, **lbl_path_style, **train_frame_configs["lbl_data_file_path_args"])
    lbl_data_file_path.grid(row=2, column=0, sticky="nsew", **train_frame_configs["lbl_data_file_path_grid"])

    btn_browse_data_file = ctk.CTkButton(master=train_frame, **btn_style, **train_frame_configs["btn_browse_data_file_args"],
                                         command=lambda : browse_file(lbl_data_file_path))
    btn_browse_data_file.grid(row=3, column=0, sticky="nsew", **train_frame_configs["btn_browse_data_file_grid"])

    # HEADING 2: feature selection
    lbl_fsel_head = ctk.CTkLabel(master=train_frame, **lbl_head_style, **train_frame_configs["lbl_fsel_head_args"], font=("Roboto", 13, "bold"))
    lbl_fsel_head.grid(row=4, column=0, sticky="nsew", **train_frame_configs["lbl_fsel_head_grid"])

    # STEP 2: select model for feature selection
    lbl_fsel_model = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_fsel_model_args"])
    lbl_fsel_model.grid(row=5, column=0, sticky="nsew", **train_frame_configs["lbl_fsel_model_grid"])

    menu_select_fsel_model = ctk.CTkOptionMenu(master=train_frame, **menu_style, values = ["Do not execute feature selection"] + fsel_model_list)
    menu_select_fsel_model.grid(row=6, column=0, sticky="nsew", **train_frame_configs["menu_select_fsel_model_grid"])

    # STEP 3: select cross-validation for feature selection
    lbl_fsel_cv = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_fsel_cv_args"])
    lbl_fsel_cv.grid(row=7, column=0, sticky="nsew", **train_frame_configs["lbl_fsel_cv_grid"])

    menu_select_fsel_cv = ctk.CTkOptionMenu(master=train_frame, **menu_style, values = ["Do not execute feature selection"] + cv_list)
    menu_select_fsel_cv.grid(row=8, column=0, sticky="nsew", **train_frame_configs["menu_select_fsel_cv_grid"])

    # HEADING 3: Model training
    lbl_train_head = ctk.CTkLabel(master=train_frame, **lbl_head_style, **train_frame_configs["lbl_train_head_args"], font=("Roboto", 13, "bold"))
    lbl_train_head.grid(row=9, column=0, sticky="nsew", **train_frame_configs["lbl_train_head_grid"])

    # STEP 4: select cross-validation for nested outer loop
    lbl_outer_cv = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_outer_cv_args"])
    lbl_outer_cv.grid(row=10, column=0, sticky="nsew", **train_frame_configs["lbl_outer_cv_grid"])

    menu_select_outer_cv = ctk.CTkOptionMenu(master=train_frame, **menu_style, values=cv_list)
    menu_select_outer_cv.grid(row=11, column=0, sticky="nsew", **train_frame_configs["menu_select_outer_cv_grid"])

    # STEP 5: select cross-validation for nested inner loop
    lbl_inner_cv = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_inner_cv_args"])
    lbl_inner_cv.grid(row=12, column=0, sticky="nsew", **train_frame_configs["lbl_inner_cv_grid"])

    menu_select_inner_cv = ctk.CTkOptionMenu(master=train_frame, **menu_style, values=cv_list)
    menu_select_inner_cv.grid(row=13, column=0, sticky="nsew", **train_frame_configs["menu_select_inner_cv_grid"])

    # STEP 6: select model for training
    lbl_train_model = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_train_model_args"])
    lbl_train_model.grid(row=14, column=0, sticky="nsew", **train_frame_configs["lbl_train_model_grid"])

    menu_select_train_model = ctk.CTkOptionMenu(master=train_frame, **menu_style, values=train_model_list)
    menu_select_train_model.grid(row=15, column=0, sticky="nsew", **train_frame_configs["menu_select_train_model_grid"])

    # STEP 7: upload param file for hyperparameter tuning
    lbl_param_file_text = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_param_file_text_args"])
    lbl_param_file_text.grid(row=16, column=0, sticky="nsew", **train_frame_configs["lbl_param_file_text_grid"])

    lbl_param_file_path = ctk.CTkLabel(master=train_frame, **lbl_path_style, **train_frame_configs["lbl_param_file_path_args"])
    lbl_param_file_path.grid(row=17, column=0, sticky="nsew", **train_frame_configs["lbl_param_file_path_grid"])

    btn_browse_param_file = ctk.CTkButton(master=train_frame, **btn_style, **train_frame_configs["btn_browse_param_file_args"],
                                          command=lambda : browse_file(lbl_param_file_path))
    btn_browse_param_file.grid(row=18, column=0, sticky="nsew", **train_frame_configs["btn_browse_param_file_grid"])

    # HEADING: saving results
    lbl_save_head = ctk.CTkLabel(master=train_frame, **lbl_head_style, **train_frame_configs["lbl_save_head_args"], font=("Roboto", 13, "bold"))
    lbl_save_head.grid(row=19, column=0, sticky="nsew", **train_frame_configs["lbl_save_head_grid"])

    # STEP 8: select directory for output storing
    lbl_out_dir_text = ctk.CTkLabel(master=train_frame, **lbl_text_style, **train_frame_configs["lbl_out_dir_text_args"])
    lbl_out_dir_text.grid(row=20, column=0, sticky="nsew", **train_frame_configs["lbl_out_dir_text_grid"])

    lbl_out_dir_path = ctk.CTkLabel(master=train_frame, **lbl_path_style, **train_frame_configs["lbl_out_dir_path_args"])
    lbl_out_dir_path.grid(row=21, column=0, sticky="nsew", **train_frame_configs["lbl_out_dir_path_grid"])

    btn_browse_out_dir = ctk.CTkButton(master=train_frame, **btn_style, **train_frame_configs["btn_browse_out_dir_args"],
                                       command=lambda : browse_folder(lbl_out_dir_path))
    btn_browse_out_dir.grid(row=22, column=0, sticky="nsew", **train_frame_configs["btn_browse_out_dir_grid"])

    sep = ttk.Separator(master=train_frame, **train_frame_configs["sep_args"])
    sep.grid(row=23, column=0, sticky="nsew", **train_frame_configs["sep_grid"])

    # STEP 9: start training phase
    btn_run = ctk.CTkButton(master=train_frame, **btn_enb_state_style, **train_frame_configs["btn_run_args"],
                            command=lambda : start_training(lbl_data_file_path.cget("text"), menu_select_fsel_model.get(), menu_select_fsel_cv.get(), menu_select_outer_cv.get(), menu_select_inner_cv.get(),
                                                            menu_select_train_model.get(), lbl_param_file_path.cget("text"), lbl_out_dir_path.cget("text"), btn_run))
    btn_run.grid(row=24, column=0, sticky="nsew", **train_frame_configs["btn_run_grid"])


def start_training(data_path, fsel_model, fsel_cv, outer_cv, inner_cv, train_model, params_path, repo_path, btn_run):
    err_msg_arr = ["No File Selected", "No Folder Selected"]
    if data_path in err_msg_arr:
        messagebox.showerror(title="Input Error", message="Please choose a valid file for data training.")
    elif fsel_model == "Do not execute feature selection" and fsel_model != fsel_cv:
        messagebox.showerror(title="Input Error", message="If feature selection is not required, the model and cross-validation fields must match.")
    elif fsel_cv == "Do not execute feature selection" and fsel_cv != fsel_model:
        messagebox.showerror(title="Input Error", message="If feature selection is not required, the model and cross-validation fields must match.")
    elif params_path in err_msg_arr:
        messagebox.showerror(title="Input Error", message="Please choose a valid file for hyperparameter tuning.")
    elif repo_path in err_msg_arr:
        messagebox.showerror(title="Input Error", message="Please choose a valid directory for storing results.")
    else:
        threading.Thread(target=exec_train_task,
                         args=(data_path, fsel_model, fsel_cv, outer_cv, inner_cv, train_model, params_path, repo_path, btn_run)).start()

