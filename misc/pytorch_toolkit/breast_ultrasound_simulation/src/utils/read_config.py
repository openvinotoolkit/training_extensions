import json

def load_config(file_path):

    with open(file_path) as f1:
        j_file = json.load(f1)
    return j_file
