"""Convert dataset: Public dataset (Jester[RawFrames], AVA) --> Datumaro dataset (CVAT)."""
import os
import os.path as osp

from lxml import etree
import cv2
import pathlib
import shutil

# classification
def convert_jester_dataset_to_datumaro(src_path, dst_path):
    ## Prepare dst_path
    frames_dir_path = osp.join(src_path, 'frames')

    phases = ['train', 'val', 'test']
    for phase in phases:
        txt_path = osp.join(src_path, '{}_list_rawframes.txt'.format(phase))

        ####
        txt = open(txt_path, 'r')
        pathlib.Path(osp.join(dst_path, phase)).mkdir(parents=True, exist_ok=True)
        for i, line in enumerate(txt.readlines()):
            video_dir, _, class_idx = line[:-1].split(' ')
            print('[*] video_dir: ', video_dir)
            print('[*] class_idx: ', class_idx)
            
            video_path = osp.join(frames_dir_path, video_dir)
            frame_list = os.listdir(video_path)
            n_frames = len(frame_list)

            shutil.copytree(video_path, osp.join(dst_path, phase, 'video_{}/images'.format(str(i)), ))
            
            annotations = etree.Element('annotations')
            
            version = etree.Element('version')
            version.text = '1.1'
            annotations.append(version)
            
            meta = etree.Element('meta')
            annotations.append(meta)

            task = etree.Element('task')
            meta.append(task)

            id = etree.Element('id')
            id.text = str(i)
            task.append(id)

            name = etree.Element('name')
            name.text = 'v{}'.format(i)
            task.append(name)

            size = etree.Element('size')
            size.text = str(n_frames) ####
            task.append(size)

            mode = etree.Element('mode')
            mode.text = 'interpolation'
            task.append(mode)

            overlap = etree.Element('overlap')
            overlap.text = '2' ####
            task.append(overlap)
            
            bugtracker = etree.Element('bugtracker')
            bugtracker.text =''
            task.append(bugtracker)

            created = etree.Element('created')
            created.text = ''
            task.append(created)

            updated = etree.Element('updated')
            updated.text = ''
            task.append(updated)

            start_frame = etree.Element('start_frame')
            start_frame.text = '0'
            task.append(start_frame)

            stop_frame = etree.Element('stop_frame')
            stop_frame.text = str(n_frames)
            task.append(stop_frame)

            frame_filter = etree.Element('frame_filter')
            frame_filter.text = '1'
            task.append(frame_filter)

            z_order = etree.Element('z_order')
            z_order.text = str(True)
            task.append(z_order)

            labels = etree.Element('labels')
            task.append(labels)

            label = etree.Element('label')
            labels.append(label)

            name = etree.Element('name')
            name.text = str(class_idx)
            label.append(name)

            attributes = etree.Element('attributes')
            attributes.text = ''
            label.append(attributes)


            segments = etree.Element('segments')
            segments.text =''
            task.append(segments)

            original_size = etree.Element('original_size')
            task.append(original_size)

            sample_frame = cv2.imread(osp.join(video_path, frame_list[0]))
            original_size_width = etree.Element('width')
            original_size_width.text = str(sample_frame.shape[1])
            original_size.append(original_size_width)

            original_size_height = etree.Element('height')
            original_size_height.text = str(sample_frame.shape[0])
            original_size.append(original_size_height)

            for j, frame in enumerate(frame_list):
                image = etree.Element(
                    'image', 
                    id=str(j), 
                    name=str(frame), 
                    width=str(sample_frame.shape[1]),
                    height=str(sample_frame.shape[0])
                )
                tag = etree.Element(
                    'tag',
                    label=str(class_idx),
                    source="manual"
                )
                tag.text=''
                image.append(tag)
                annotations.append(image)
            
            et = etree.ElementTree(annotations)
            et.write(osp.join(dst_path, phase, 'video_{}/annotations.xml'.format(str(i))), pretty_print=True, xml_declaration=True, encoding="utf-8")

# detection
def convert_ava_dataset(path):
    #TODO: make it more general
    pass


def main(src_path, dst_path):
    convert_jester_dataset_to_datumaro(src_path, dst_path)

if __name__ == '__main__':
    main('/local/sungmanc/datasets/jester_SC', '/local/sungmanc/datasets/jester_SC_cvat_multifolder_classification')