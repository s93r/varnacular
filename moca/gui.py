import os

import customtkinter as ctk

from catalog_datasets import open_datasets_catalog_window
from catalog_models import open_models_catalog_window
from prediction_module import open_prediction_window
from training_module import open_training_window
from utils import load_json

main_frame_configs_file_path = os.path.join(os.path.dirname(__file__), 'ui_settings', 'settings_main_frame.json')
main_frame_configs = load_json(main_frame_configs_file_path)
widgets_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_widgets.json")
widgets_configs = load_json(widgets_configs_file_path)

btn_style = widgets_configs["btn_style"]

def main():
    main_frame = ctk.CTk(main_frame_configs["main_frame_style"]["fg_color"])
    main_frame.title(main_frame_configs["main_frame_style"]["title"])
    main_frame.iconbitmap("images/logo_moca.ico")
    main_frame.geometry(main_frame_configs["main_frame_style"]["geometry"])
    main_frame.resizable(main_frame_configs["main_frame_style"]["resizable"]["width"],
                         main_frame_configs["main_frame_style"]["resizable"]["height"])

    btn_open_train_frame = ctk.CTkButton(master=main_frame, **btn_style, **main_frame_configs["btn_open_train_frame_args"],
                                         command=lambda : open_training_window(main_frame))
    btn_open_train_frame.grid(row=0, column=0, sticky="nsew", **main_frame_configs["btn_open_train_frame_grid"])

    btn_open_test_frame = ctk.CTkButton(master=main_frame, **btn_style, **main_frame_configs["btn_open_pred_frame_args"],
                                        command=lambda : open_prediction_window(main_frame))
    btn_open_test_frame.grid(row=1, column=0, sticky="nsew", **main_frame_configs["btn_open_pred_frame_grid"])

    btn_open_models_catalog = ctk.CTkButton(master=main_frame, **btn_style, **main_frame_configs["btn_open_models_catalog_args"],
                                           command=lambda : open_models_catalog_window())
    btn_open_models_catalog.grid(row=2, column=0, sticky="nsew", **main_frame_configs["btn_open_models_catalog_grid"])

    btn_open_datasets_catalog = ctk.CTkButton(master=main_frame, **btn_style, **main_frame_configs["btn_open_datasets_catalog_args"],
                                             command=lambda : open_datasets_catalog_window())
    btn_open_datasets_catalog.grid(row=3, column=0, sticky="nsew", **main_frame_configs["btn_open_datasets_catalog_grid"])

    main_frame.grid_rowconfigure(index=0, weight=1)
    main_frame.grid_rowconfigure(index=1, weight=1)
    main_frame.grid_rowconfigure(index=2, weight=1)
    main_frame.grid_rowconfigure(index=3, weight=1)
    main_frame.grid_columnconfigure(index=0, weight=1)

    main_frame.mainloop()

if __name__ == "__main__":
    main()