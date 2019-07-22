import argparse

import cv2
import os

from common import fit_to_max_size

class CvatAnnotation:
    def __init__(self, path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()

        self.annotation = {}

        self.task_name = root.find('meta').find('task').find('name').text

        # id_map = {}

        # for imname in os.listdir('/home/ikrylov/Desktop/patterns_20190607/Full_size'):
        #     pattern_index, pattern_name = imname.split('_')
        #     pattern_name = pattern_name[:-8]
        #
        #     id_map[pattern_index] = pattern_name

        for track in root.findall('track'):
            for box in track.findall('box'):
                frame = int(box.get('frame'))

                xtl = int(float(box.get('xtl')))
                ytl = int(float(box.get('ytl')))
                xbr = int(float(box.get('xbr')))
                ybr = int(float(box.get('ybr')))

                pattern_id = box.find('attribute')

                # try:
                #     pattern_id.text = id_map[pattern_id.text]
                # except:
                #     pass

                if pattern_id is None:
                    pattern_id = ''
                else:
                    pattern_id = pattern_id.text

                if frame not in self.annotation:
                    self.annotation[frame] = []

                self.annotation[frame].append({
                    'xtl': xtl,
                    'ytl': ytl,
                    'xbr': xbr,
                    'ybr': ybr,
                    'id': pattern_id
                })

        #tree.write(path, short_empty_elements=True)


if __name__ == '__main__':

    def parse_args():
        args = argparse.ArgumentParser()
        args.add_argument('--path', required=True)
        args.add_argument('--videos_root', default='/home/ikrylov/datasets/')
        args.add_argument('--crops_folder', required=False)
        args.add_argument('--patterns_root', default='/home/ikrylov/datasets/textile/20190523/img_retrieval_cal_acc/query_db')

        return args.parse_args()

    args = parse_args()

    annotation = CvatAnnotation(args.path)
    capture = cv2.VideoCapture(os.path.join(args.videos_root, annotation.task_name))


    if args.crops_folder:
        os.makedirs(args.crops_folder, exist_ok=True)

    idx = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        if idx % 100 != 0:
            idx += 1
            continue

        if idx in annotation.annotation:
            for obj in annotation.annotation[idx]:

                if args.crops_folder:
                    os.makedirs(os.path.join(args.crops_folder, obj['id']), exist_ok=True)
                    cv2.imwrite(os.path.join(args.crops_folder, obj['id'], 'frame_{}.png'.format(idx)),
                                frame[obj['ytl']:obj['ybr'], obj['xtl']:obj['xbr']])

                cv2.rectangle(frame, (obj['xtl'], obj['ytl']), (obj['xbr'], obj['ybr']),
                              (0, 255, 0), 5)
                cv2.putText(frame,  obj['id'], (obj['xtl'], obj['ytl'] + 100), 1, 5.0, (255, 0, 0), 5)

        frame = cv2.resize(frame, (1280, 720))

        if obj['id']:
            pattern = cv2.imread(os.path.join(args.patterns_root, obj['id'] + '.png'))

            if pattern is None:
                print(os.path.join(args.patterns_root, obj['id'] + '.png'))

            pattern = fit_to_max_size(pattern, 256)


            frame[:pattern.shape[0], :pattern.shape[1]] = pattern

        cv2.imshow('Webcam', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        idx += 1
