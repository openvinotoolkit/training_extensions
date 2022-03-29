import json
import os.path as osp
from io import BytesIO
from zipfile import ZipFile


def update_stage():
    """ update stage recipe
    """
    pass


def read_label_schema(path, name=True):
    serialized_label_schema = []
    if any(path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
        label_schema_path = osp.join(osp.dirname(path), "label_schema.json")
        if osp.exists(label_schema_path):
            with open(
                osp.join(osp.dirname(path), "label_schema.json"), encoding="UTF-8"
            ) as read_file:
                serialized_label_schema = json.load(read_file)
    elif path.endswith(".zip"):
        with ZipFile(path) as read_zip_file:
            zfiledata = BytesIO(
                read_zip_file.read(
                    osp.join("python", "demo_package-0.0-py3-none-any.whl")
                )
            )
            with ZipFile(zfiledata) as read_whl_file:
                with read_whl_file.open(
                    osp.join("demo_package", "config.json")
                ) as read_file:
                    serialized_label_schema = json.load(read_file)["model_parameters"][
                        "labels"
                    ]
    if serialized_label_schema:
        if name:
            all_classes = [labels['name'] for labels in serialized_label_schema['all_labels'].values()]
        else:
            all_classes = serialized_label_schema
    else:
        all_classes = []
    return all_classes


def read_binary(path):
    """
    Loads binary data stored at path.

        Args:
            path: A path where to load data from.
    """

    with open(path, "rb") as read_file:
        return read_file.read()
