import os
import shutil
import time
import traceback
from tkinter import messagebox

import joblib
import  pandas as pd

from utils import load_json, create_external_folder


widgets_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_widgets.json")
widgets_configs = load_json(widgets_configs_file_path)

btn_enb_state_style = widgets_configs["btn_enb_state_style"]
btn_dsb_state_style = widgets_configs["btn_dsb_state_style"]


def exec_pred_task(data_path, refs_data_path, refs_model_path, out_dir_path, btn_run):
    btn_run.configure(**btn_dsb_state_style)
    predict_data(data_path, refs_data_path, refs_model_path, out_dir_path)
    btn_run.configure(**btn_enb_state_style)


def predict_data(data_path, refs_data_path, refs_model_path, dir_path):

    try:
        res_dir = create_external_folder(dir_base=dir_path, dir_name=f"ml_proj_pred_{time.strftime("%Y%m%d_%H%M%S")}")
    except Exception as e:
        messagebox.showerror(title="Prediction Error", message=f"An error occurred when creating the result folder: {e}.", detail=traceback.format_exc())
        return

    try:
        df = load_data(data_path)
    except Exception as e:
        messagebox.showerror(title="Prediction Error", message=f"An error occurred when loading the data file: {e}.", detail=traceback.format_exc())
        return

    model = load_model(refs_model_path)

    try:
        probs = predict_probs(model, df)
        preds = predict_class(model, df)
    except Exception as e:
        messagebox.showerror(title="Prediction Error", message=f"An error occurred when performing the prediction: {e}.", detail=traceback.format_exc())
        return

    prob_df = pd.DataFrame(probs, columns=model.classes_).map(lambda x : f"{x:.3f}")
    prob_df.index += 1
    prob_df.reset_index(inplace=True)

    pred_df = df.assign(label=preds)

    export_dataframe(df=prob_df, path=res_dir, basename="results_probs", extension="xlsx")
    export_dataframe(df=pred_df, path=res_dir, basename="results_preds", extension="xlsx")
    export_dataframe(df=prob_df, path=res_dir, basename="results_probs", extension="csv")
    export_dataframe(df=pred_df, path=res_dir, basename="results_preds", extension="csv")

    shutil.copy(data_path, res_dir)

    out_refs_dir = create_external_folder(res_dir, "reference_src")
    shutil.copy(refs_data_path, out_refs_dir)
    shutil.copy(refs_model_path, out_refs_dir)


def load_model(file_path):
    model_pkg = joblib.load(file_path)
    model = model_pkg["model"]
    return model


def load_data(file_path):
    data = pd.read_excel(file_path)
    return data


def predict_probs(model, data):
    probs = model.predict_proba(data)
    return probs


def predict_class(model, data):
    preds = model.predict(data)
    return preds


def export_dataframe(df, path, basename, extension):
    match extension:
        case 'csv':
            export_dataframe_to_csv(df, path, basename)
        case 'xlsx':
            export_dataframe_to_excel(df, path, basename)


def export_dataframe_to_excel(df, path, basename):
    df.to_excel(f"{path}/{basename}.xlsx", index=False)


def export_dataframe_to_csv(df, path, basename):
    df.to_csv(f"{path}/{basename}.csv", index=False)

