"""Convert dataset: Public dataset (Jester[RawFrames], AVA) --> Datumaro dataset (CVAT).

This script contains a lot of hardcoding to create an.xml file that Datumaro can consume.

Current Datumaro format for video (CVAT)

::

    root
    |- video_0
    |    |- images
    |       |- frames_001.png
    |    |- frames_002.png
    |    |- annotations.xml
    |- video_1
    |    |- images
    |    |- annotations.xml
    |- video_2

"""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=too-many-locals, c-extension-no-member, invalid-name, too-many-statements
import argparse
import csv
import os
import os.path as osp
import pathlib
import shutil
from typing import List

import cv2

# disable B410 import_lxml - the library used only for the xml writing
from lxml import etree  # nosec B410
from tqdm import tqdm


def generate_default_cvat_xml_fields(i, video_path, frame_list):
    """Generate default CVAT xml fields required to make multi-video CVAT format by using Jester, and AVA dataset."""
    n_frames = len(frame_list)
    annotations = etree.Element("annotations")

    version = etree.Element("version")
    version.text = "1.1"
    annotations.append(version)

    meta = etree.Element("meta")
    annotations.append(meta)

    task = etree.Element("task")
    meta.append(task)

    _id = etree.Element("id")
    _id.text = str(i)
    task.append(_id)

    name = etree.Element("name")
    name.text = f"v{i}"
    task.append(name)

    size = etree.Element("size")
    size.text = str(n_frames)
    task.append(size)

    mode = etree.Element("mode")
    mode.text = "interpolation"
    task.append(mode)

    overlap = etree.Element("overlap")
    overlap.text = "2"
    task.append(overlap)

    bugtracker = etree.Element("bugtracker")
    bugtracker.text = ""
    task.append(bugtracker)

    created = etree.Element("created")
    created.text = ""
    task.append(created)

    updated = etree.Element("updated")
    updated.text = ""
    task.append(updated)

    start_frame = etree.Element("start_frame")
    start_frame.text = "0"
    task.append(start_frame)

    stop_frame = etree.Element("stop_frame")
    stop_frame.text = str(n_frames - 1)
    task.append(stop_frame)

    frame_filter = etree.Element("frame_filter")
    frame_filter.text = "1"
    task.append(frame_filter)

    z_order = etree.Element("z_order")
    z_order.text = str(True)
    task.append(z_order)

    labels = etree.Element("labels")
    task.append(labels)

    segments = etree.Element("segments")
    segments.text = ""
    task.append(segments)

    original_size = etree.Element("original_size")
    task.append(original_size)

    sample_frame = cv2.imread(osp.join(video_path, frame_list[0]))
    original_size_width = etree.Element("width")
    original_size_width.text = str(sample_frame.shape[1])
    original_size.append(original_size_width)

    original_size_height = etree.Element("height")
    original_size_height.text = str(sample_frame.shape[0])
    original_size.append(original_size_height)

    return annotations, sample_frame.shape, labels


# classification
def convert_action_cls_dataset_to_datumaro(src_path: str, dst_path: str, ann_file: str, label_map=None):
    """Convert a public dataset to multi-video CVAT (Datumaro) format.

    Supported datasets are: Jester, HMDB51, UCF101

    Args:
        src_path (str): The path to the directory containing the video files as rawframe folder.
        dst_path (str): The path to the directory where the multi-video CVAT (Datumaro) format dataset will be saved.
        ann_file (str): The path to the file containing the annotations for the videos.
        label_map (optional): The path to the file containing the mapping between class IDs and class names.

    Returns:
        None

    Examples:
        src_path = "./data/hmdb51/rawframes"
        dst_path = "./data/hmdb51/CVAT/train"
        ann_file = "./data/hmdb51/hmdb51_train_split_1_rawframes.txt"
        label_map = "./data/hmdb51/label_map.txt"
        convert_public_dataset_to_datumaro(src_path, dst_path, ann_file, label_map=label_map)
    """

    # Load the annotations file. Annotation is supposed as whitespace separated format: Video_name num_frames class_idx
    # Ex) kiss/American_History_X_kiss_h_cm_np2_le_goo_40 69 22
    with open(ann_file, "r", encoding="utf-8") as anns:
        pathlib.Path(osp.join(dst_path)).mkdir(parents=True, exist_ok=True)

        lines = anns.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            if line[0] == "#":
                continue

            # Parse the video directory and class ID from the annotations file
            video_dir, _, class_idx = line[:-1].split(" ")

            if label_map is not None:
                # Load label mapping file : brush_hair\ncarwheel\ncatch, ...
                with open(label_map, "r", encoding="utf-8") as f:
                    label_names = f.read().splitlines()
                label_mapping_dict = {str(idx): label_name for idx, label_name in enumerate(label_names)}
                class_name = label_mapping_dict[class_idx]
            else:
                class_name = class_idx

            # Prepare the output directories and file names
            video_path = osp.join(src_path, video_dir)
            video_name = f"Video_{i}"
            images_dir = osp.join(dst_path, f"{video_name}/images")

            # List the frames in the video and sort them
            frame_list = os.listdir(video_path)
            frame_list.sort()

            # Generate default CVAT XML fields for the video annotation
            annotations, img_shape, labels = generate_default_cvat_xml_fields(i, video_path, frame_list)

            # Add the video label to the annotations
            label = etree.Element("label")
            labels.append(label)

            name = etree.Element("name")
            name.text = class_name
            label.append(name)

            attributes = etree.Element("attributes")
            attributes.text = ""
            label.append(attributes)

            # Copy the video frames to the output directory and create the image tags in the annotations
            for j, frame in enumerate(frame_list):
                if not osp.exists(images_dir):
                    os.makedirs(images_dir, exist_ok=True)

                image_name = f"{j+1:05d}.jpg"
                shutil.copy(osp.join(video_path, frame), osp.join(images_dir, image_name))
                image = etree.Element(
                    "image", id=str(j), name=image_name, width=str(img_shape[1]), height=str(img_shape[0])
                )
                tag = etree.Element("tag", label=class_name, source="manual")
                tag.text = ""
                image.append(tag)
                annotations.append(image)

            et = etree.ElementTree(annotations)
            et.write(
                osp.join(dst_path, f"{video_name}/annotations.xml"),
                pretty_print=True,
                xml_declaration=True,
                encoding="utf-8",
            )


def convert_ava_dataset_to_datumaro(src_path: str, dst_path: str, ann_file: str):
    """Convert AVA dataset to multi-video CVAT (Datumaro) format.

    Args:
        src_path (str): The path to the directory containing the video files as rawframe folder.
        dst_path (str): The path to the directory where the multi-video CVAT (Datumaro) format dataset will be saved.
        ann_file (str): The path to the file containing the annotations for the videos.

    Returns:
        None

    Examples:
        src_path = "./data/ava_dataset/frames"
        dst_path = "./data/ava_dataset/CVAT/train"
        ann_file = "./data/ava_dataset/annotations/train.csv"
        convert_ava_dataset_to_datumaro(src_path, dst_path, ann_file)
    """

    video_dict = {}
    video_idx = 0
    annot_info = read_ava_csv(ann_file)
    video_dirs = os.listdir(src_path)
    video_dirs.sort()
    for video_dir in video_dirs:
        if video_dir not in annot_info:
            continue
        video_path = osp.join(src_path, video_dir)
        frame_list = os.listdir(video_path)
        frame_list.sort()
        if video_dir not in video_dict:
            annotations, _, labels = generate_default_cvat_xml_fields(video_idx, video_path, frame_list)
            label_list: List[str] = []
            video_dict[video_dir] = {"annotations": annotations, "labels": labels, "label_list": label_list}
        for frame in frame_list:
            frame_idx = int(frame.split(".")[0].split("_")[-1])
            if frame_idx in annot_info[video_dir]:
                annots = annot_info[video_dir][frame_idx]
            else:
                annots = [[0, 0, 0, 0, "EmptyFrame"]]

            for annot in annots:
                bboxes = annot[:4]
                class_idx = annot[4]
                track = etree.Element("track", id=str(video_idx), label=str(class_idx))
                if class_idx not in video_dict[video_dir]["label_list"]:
                    video_dict[video_dir]["label_list"].append(class_idx)

                    label = etree.Element("label")
                    video_dict[video_dir]["labels"].append(label)

                    name = etree.Element("name")
                    name.text = str(class_idx)
                    label.append(name)

                    attributes = etree.Element("attributes")
                    attributes.text = ""
                    label.append(attributes)
                box = etree.Element(
                    "box",
                    frame=str(frame_idx),
                    outside="0",  # not used
                    occluded="0",  # not used
                    xtl=str(bboxes[0]),
                    ytl=str(bboxes[1]),
                    xbr=str(bboxes[2]),
                    ybr=str(bboxes[3]),
                    z_order="0",
                    label=str(class_idx),
                )
                box.text = ""
                track.append(box)
                video_dict[video_dir]["annotations"].append(track)

    for (video_dir, info) in video_dict.items():
        video_path = osp.join(src_path, video_dir)
        shutil.copytree(video_path, osp.join(dst_path, f"{video_dir}/images"), copy_function=rename_and_copy)
        et = etree.ElementTree(info["annotations"])
        et.write(
            osp.join(dst_path, f"{video_dir}/annotations.xml"),
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        )


def rename_and_copy(_src, _dst):
    """Change frame name to cvat format."""
    img_name = _dst.rsplit("/", maxsplit=1)[-1]
    # FIXME This only support AVA dataset name
    if "_" in img_name:
        frame_index = int(img_name.split(".")[0].split("_")[-1])
    else:
        frame_index = int(img_name.split(".")[0])
    new_img_name = f"frame_{frame_index:06d}.png"
    _dst = _dst.replace(img_name, new_img_name)
    shutil.copy2(_src, _dst)


def read_ava_csv(csv_path):
    """Read ava format annotation csv file."""
    annot_info = {}  # {video_id: {frame_idx: [annot0, annot1, ..., annotN]}}
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for line in csv_reader:
            video_id, frame_idx, bboxes, class_idx = line[0], line[1], line[2:6], line[6]
            frame_idx = int(frame_idx)
            bboxes.append(class_idx)
            if video_id in annot_info:
                if frame_idx in annot_info[video_id]:
                    annot_info[video_id][frame_idx].append(bboxes)
                else:
                    annot_info[video_id][frame_idx] = [bboxes]
            else:
                annot_info[video_id] = {frame_idx: [bboxes]}
    return annot_info


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--src_path", required=True, type=str)
    parser.add_argument("--dst_path", required=True, type=str)
    parser.add_argument("--ann_file", required=True, type=str)
    parser.add_argument("--label_map", type=str, default=None)
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if args.task == "action_classification":
        convert_action_cls_dataset_to_datumaro(args.src_path, args.dst_path, args.ann_file, args.label_map)
    elif args.task == "action_detection":
        convert_ava_dataset_to_datumaro(args.src_path, args.dst_path, args.ann_file)


if __name__ == "__main__":
    main()
