import os
import shutil
import statistics
import time
import traceback
from tkinter import messagebox

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text, plot_tree

from tools import get_model_code_dict, get_cv_dict, get_model_dict
from utils import create_external_folder, export_strarr_to_file, load_json

widgets_configs_file_path = os.path.join(os.path.dirname(__file__), "ui_settings", "settings_widgets.json")
widgets_configs = load_json(widgets_configs_file_path)

btn_enb_state_style = widgets_configs["btn_enb_state_style"]
btn_dsb_state_style = widgets_configs["btn_dsb_state_style"]

cv_dict = get_cv_dict()
model_dict = get_model_dict()
code_dict = get_model_code_dict()


def exec_train_task(data_path, fsel_model, fsel_cv, outer_loop, inner_loop, train_model, param_path, dir_path, btn_run):
    btn_run.configure(**btn_dsb_state_style)
    train_data(data_path, fsel_model, fsel_cv, outer_loop, inner_loop, train_model, param_path, dir_path)
    btn_run.configure(**btn_enb_state_style)


def train_data(data_path, fsel_model, fsel_cv, outer_loop, inner_loop, train_model, param_path, dir_path):

    try:
        res_dir = create_external_folder(dir_base=dir_path, dir_name=f"ml_proj_train_{time.strftime("%Y%m%d_%H%M%S")}")
    except Exception as e:
        messagebox.showerror(title="Training Error", message=f"An error occurred when creating the result folder: {e}.", detail=traceback.format_exc())
        return

    try:
        df, X, y, feature_names = load_dataframe_from_file(file_path=data_path)
    except Exception as e:
        messagebox.showerror(title="Training Error", message=f"An error occurred when loading the data file: {e}.", detail=traceback.format_exc())
        return

    no_fsel = "Do not execute feature selection"

    if fsel_model != no_fsel and fsel_cv != no_fsel:
        rfe_model = model_dict[fsel_model]
        rfe_cv = cv_dict[fsel_cv]

        best_mask = select_features(X, y, rfe_model, rfe_cv)
        best_cols = X.columns[best_mask]
        drop_cols = [feature for feature in feature_names if feature not in best_cols.tolist()]

        df = df.drop(drop_cols, axis="columns")
        X = df.iloc[:, 1:]
        y = df.iloc[:, :1]

    outer_cv = cv_dict[outer_loop]
    inner_cv = cv_dict[inner_loop]

    model = model_dict[train_model]
    code = code_dict[train_model]

    try:
        param_grid = load_paramgrid_from_file(file_path=param_path)
    except Exception as e:
        messagebox.showerror(title="Training Error", message=f"An error occurred when loading the parameter file: {e}", detail=traceback.format_exc())
        return

    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "balanced": make_scorer(balanced_accuracy_score),
        "mcc": make_scorer(matthews_corrcoef),
    }

    outer_accuracy = []
    outer_balanced = []
    outer_mcc = []

    best_model = None
    best_accuracy = float("-inf")

    try:
        for train_index, test_index in outer_cv.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring=scorers,
                                       refit="accuracy")
            grid_search.fit(X_train, y_train.values.ravel())

            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_test)

            tmp_accuracy = accuracy_score(y_test, y_pred)
            tmp_balanced = balanced_accuracy_score(y_test, y_pred)
            tmp_mcc = matthews_corrcoef(y_test, y_pred)

            outer_accuracy.append(tmp_accuracy)
            outer_balanced.append(tmp_balanced)
            outer_mcc.append(tmp_mcc)

            if tmp_accuracy > best_accuracy:
                best_accuracy = tmp_accuracy
                best_model = best_estimator

    except Exception as e:
        messagebox.showerror(title="Training Error", message=f"An error occurred when training the model: {e}", detail=traceback.format_exc())
        return

    avg_accuracy = "Nested CV standard accuracy: %.3f +/- %.3f" % (statistics.mean(outer_accuracy), statistics.stdev(outer_accuracy))
    avg_balanced = "Nested CV balanced accuracy: %.3f +/- %.3f" % (statistics.mean(outer_balanced), statistics.stdev(outer_balanced))
    avg_mcc = "Nested CV matthews correlation coefficient: %.3f +/- %.3f" % (statistics.mean(outer_mcc), statistics.stdev(outer_mcc))

    if code == "CART":
        export_tree_graph_to_file(model=best_model, path=res_dir, basename="tree_plot", extension="png")
        export_tree_trace_to_file(model=best_model, path=res_dir, basename="tree_text", extension="txt")

    y_pred = best_model.predict(X)

    pred_accuracy = accuracy_score(y, y_pred)
    pred_balanced = balanced_accuracy_score(y, y_pred)
    pred_mcc = matthews_corrcoef(y, y_pred)

    accuracy = f"Best model standard accuracy: {pred_accuracy:.3f}"
    balanced = f"Best model balanced accuracy: {pred_balanced:.3f}"
    mcc = f"Best model matthews correlation coefficient: {pred_mcc:.3f}"

    export_strarr_to_file(strarr=[avg_accuracy, avg_balanced, avg_mcc, accuracy, balanced, mcc], path=res_dir, basename="metrics", extension="txt")

    conf_matr = confusion_matrix(y, y_pred, labels=best_model.classes_)
    export_confusion_matrix_to_file(conf_matr=conf_matr, model=best_model, path=res_dir, basename="confusion_matrix", extension="png")

    inner_repo_path_model = "models"
    inner_repo_path_data = "datasets"

    shutil.copy(data_path, inner_repo_path_data)
    shutil.copy(data_path, res_dir)

    model_info_dict = {"model": best_model, "fsel_model": fsel_model, "fsel_cv": fsel_cv, "outer_cv": outer_loop, "inner_cv": inner_loop, "train_model": train_model}
    export_model_info_to_file(info_dict=model_info_dict, path=res_dir, basename="model_info", extension="txt")

    export_model_to_file(model=best_model, model_code=code, data_path=data_path, outer_repo_path=res_dir)
    save_model_copy_to_repo(model=best_model, model_code=code, data_path=data_path, inner_repo_path=inner_repo_path_model)


def load_dataframe_from_file(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, 1:]
    y = df.iloc[:, :1]
    feature_names = df.columns[1:]
    return df, X, y, feature_names


def load_paramgrid_from_file(file_path):
    param_grid = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            param_grid[key] = eval(value)
    return param_grid


def export_tree_graph_to_file(model, path, basename, extension):
    plt.figure(figsize=(200, 125))
    plot_tree(model, feature_names=model.feature_names_in_, class_names=model.classes_, filled=True, rounded=True)
    plt.savefig(os.path.join(path, f"{basename}.{extension}"))
    plt.close()


def export_tree_trace_to_file(model, path, basename, extension):
    tree_text = export_text(model, feature_names=model.feature_names_in_)
    with open(os.path.join(path, f"{basename}.{extension}"), 'w') as f:
        f.write(tree_text)


def select_features(X, y, estimator, cv):
    rfecv = RFECV(estimator=estimator, cv=cv, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X, y.values.ravel())
    return rfecv.support_


def export_model_info_to_file(info_dict, path, basename, extension):
    model = info_dict["model"]
    fsel_model = info_dict["fsel_model"]
    train_model = info_dict["train_model"]
    fsel_cv = info_dict["fsel_cv"]
    outer_cv = info_dict["outer_cv"]
    inner_cv = info_dict["inner_cv"]

    with open(os.path.join(path, f"{basename}.{extension}"), 'w') as file:
        file.write(f"FEATURE SELECTION MODEL:\n")
        file.write(f"{fsel_model}\n\n")
        file.write(f"FEATURE SELECTION CROSS-VALIDATION:\n")
        file.write(f"{fsel_cv}\n\n")
        file.write(f"TRAINING MODEL:\n")
        file.write(f"{train_model}\n\n")
        file.write(f"TRAINING MODEL NESTED CROSS-VALIDATION OUTER LOOP:\n")
        file.write(f"{outer_cv}\n\n")
        file.write(f"TRAINING MODEL NESTED CROSS-VALIDATION INNER LOOP:\n")
        file.write(f"{inner_cv}\n\n")
        file.write(f"TRAINING MODEL HYPERPARAMETERS:\n")
        file.write(f"{model.get_params()}\n\n")
        file.write(f"SELECTED FEATURES:\n")
        for feature in model.feature_names_in_:
            file.write(f"{feature}\n")


def export_model_to_file(model, model_code, data_path, outer_repo_path):
    data_basename = os.path.splitext(os.path.basename(data_path))[0]
    filename = f"pretrained_{data_basename}_{model_code}.pkl"
    outer_path = os.path.join(outer_repo_path, filename)
    model_pkg = {"model": model, "data": data_path}
    joblib.dump(model_pkg, outer_path)


def save_model_copy_to_repo(model, model_code, data_path, inner_repo_path):
    data_basename = os.path.splitext(os.path.basename(data_path))[0]
    filename = f"pretrained_{data_basename}_{model_code}.pkl"
    inner_path = os.path.join(inner_repo_path, filename)
    model_pkg = {"model": model, "data": data_path}
    joblib.dump(model_pkg, inner_path)


def export_confusion_matrix_to_file(conf_matr, model, path, basename, extension):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matr, cmap="Blues", annot=True, fmt="d", cbar=True, xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{basename}.{extension}"))
    plt.close()

