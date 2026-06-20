import json
import os


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def create_external_folder(dir_base, dir_name):
    dir_path = os.path.join(dir_base, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def export_strarr_to_file(strarr, path, basename, extension):
    with open(os.path.join(path, f"{basename}.{extension}"), 'w') as file:
        file.write("\n".join(strarr))