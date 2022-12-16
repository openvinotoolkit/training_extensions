"""Convert dataset: Public dataset (Jester[RawFrames], AVA) --> Datumaro dataset (CVAT).

It contains lots of hard-coded to make .xml file consumed on Datumaro.

Current Datumaro format for video (CVAT)

root
|- video_0
    |- images
        |- frames_001.png
        |- frames_002.png
    |- annotations.xml
|- video_1
    |- images
    |- annotations.xml
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

import csv
import os
import os.path as osp
import pathlib
import shutil

import cv2
from lxml import etree


def generate_default_cvat_xml_fields(i, class_idx, video_path, frame_list):
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
    size.text = str(n_frames)  ####
    task.append(size)

    mode = etree.Element("mode")
    mode.text = "interpolation"
    task.append(mode)

    overlap = etree.Element("overlap")
    overlap.text = "2"  ####
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
    stop_frame.text = str(n_frames)
    task.append(stop_frame)

    frame_filter = etree.Element("frame_filter")
    frame_filter.text = "1"
    task.append(frame_filter)

    z_order = etree.Element("z_order")
    z_order.text = str(True)
    task.append(z_order)

    labels = etree.Element("labels")
    task.append(labels)

    label = etree.Element("label")
    labels.append(label)

    name = etree.Element("name")
    name.text = str(class_idx)
    label.append(name)

    attributes = etree.Element("attributes")
    attributes.text = ""
    label.append(attributes)

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

    return annotations, sample_frame.shape


# classification
def convert_jester_dataset_to_datumaro(src_path, dst_path):
    """Convert Jester dataset to multi-video CVAT (Datumaro) format."""
    ## Prepare dst_path
    frames_dir_path = osp.join(src_path, "frames")

    phases = ["train", "val", "test"]
    for phase in phases:
        txt_path = osp.join(src_path, f"{phase}_list_rawframes.txt")
        with open(txt_path, "r", encoding="utf-8") as txt:
            pathlib.Path(osp.join(dst_path, phase)).mkdir(parents=True, exist_ok=True)

            for i, line in enumerate(txt.readlines()):
                video_dir, _, class_idx = line[:-1].split(" ")

                video_path = osp.join(frames_dir_path, video_dir)
                frame_list = os.listdir(video_path)

                shutil.copytree(
                    video_path,
                    osp.join(
                        dst_path,
                        phase,
                        f"video_{i}/images",
                    ),
                )

                annotations, img_shape = generate_default_cvat_xml_fields(i, class_idx, video_path, frame_list)

                for j, frame in enumerate(frame_list):
                    image = etree.Element(
                        "image", id=str(j), name=str(frame), width=str(img_shape[1]), height=str(img_shape[0])
                    )
                    tag = etree.Element("tag", label=str(class_idx), source="manual")
                    tag.text = ""
                    image.append(tag)
                    annotations.append(image)

                et = etree.ElementTree(annotations)
                et.write(
                    osp.join(dst_path, phase, f"video_{i}/annotations.xml"),
                    pretty_print=True,
                    xml_declaration=True,
                    encoding="utf-8",
                )


def convert_ava_dataset_to_datumaro(src_path, dst_path):
    """Convert AVA dataset to multi-video CVAT (Datumaro) format."""
    ## Prepare dst_path
    frames_dir_path = osp.join(src_path, "frames")

    phases = ["train", "valid"]
    for phase in phases:
        csv_path = osp.join(src_path, f"annotations/{phase}.csv")
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            pathlib.Path(osp.join(dst_path, phase)).mkdir(parents=True, exist_ok=True)

            video_dict = {}
            for i, line in enumerate(csv_reader):
                video_id, frame_idx, bboxes, class_idx = line[0], line[1], line[2:6], line[6]
                video_path = osp.join(frames_dir_path, video_id)
                frame_list = os.listdir(video_path)

                if video_id not in video_dict:
                    annotations, _ = generate_default_cvat_xml_fields(i, class_idx, video_path, frame_list)
                    track = etree.Element("track", id=str(i), label=str(class_idx))
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
                    )
                    box.text = ""
                    track.append(box)
                    annotations.append(track)
                    video_dict[video_id] = annotations
                else:
                    track = etree.Element("track", id=str(i), label=str(class_idx))
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
                    )
                    box.text = ""
                    track.append(box)
                    video_dict[video_id].append(track)

            for i, (_, v) in enumerate(video_dict.items()):
                shutil.copytree(video_path, osp.join(dst_path, phase, f"video_{i}/images"))
                et = etree.ElementTree(v)
                et.write(
                    osp.join(dst_path, phase, f"video_{i}/annotations.xml"),
                    pretty_print=True,
                    xml_declaration=True,
                    encoding="utf-8",
                )


def main(src_path, dst_path):
    """Main function."""
    # convert_jester_dataset_to_datumaro(src_path, dst_path)
    convert_ava_dataset_to_datumaro(src_path, dst_path)


if __name__ == "__main__":
    main("/local/sungmanc/datasets/ava_SC", "/local/sungmanc/datasets/ava_SC_cvat_multifolder_detection")
