"""
This script is slightly modified for facial landmark localization.
You can find the original code here:
https://github.com/opencv/opencv/blob/master/samples/dnn/resnet_ssd_face_python.py
"""
import cv2 as cv
from cv2 import dnn

WIDTH = 300
HEIGHT = 300

PROTOTXT = '/home/vic/workspace/facial_landmark/openvino_training_extensions/pytorch_toolkit/face_recognition/utils/assets/deploy.prototxt'
MODEL = '/home/vic/workspace/facial_landmark/openvino_training_extensions/pytorch_toolkit/face_recognition/utils/assets/res10_300x300_ssd_iter_140000.caffemodel'

# CASCADES_FILE = "/opt/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml"
# CASCADES = cv.CascadeClassifier(CASCADES_FILE)

NET = dnn.readNetFromCaffe(PROTOTXT, MODEL)
# VIDEO = '/home/robin/Documents/landmark/dataset/300VW_Dataset_2015_12_14/538/vid.avi'


def get_facebox(image=None, threshold=0.5):
    """
    Get the bounding box of faces in image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    confidences = []
    faceboxes = []

    NET.setInput(dnn.blobFromImage(
        image, 1.0, (WIDTH, HEIGHT), (104.0, 177.0, 123.0), False, False))
    detections = NET.forward()

    for result in detections[0, 0, :, :]:
        confidence = result[2]
        if confidence > threshold:
            x_left_bottom = int(result[3] * cols)
            y_left_bottom = int(result[4] * rows)
            x_right_top = int(result[5] * cols)
            y_right_top = int(result[6] * rows)
            confidences.append(confidence)
            faceboxes.append(
                [x_left_bottom, y_left_bottom, x_right_top, y_right_top])
    return confidences, faceboxes


def draw_result(image, confidences, faceboxes):
    """Draw the detection result on image"""
    for result in zip(confidences, faceboxes):
        conf = result[0]
        facebox = result[1]

        cv.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), (0, 255, 0))
        label = "face: %.4f" % conf
        label_size, base_line = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                     (facebox[0] + label_size[0],
                      facebox[1] + base_line),
                     (0, 255, 0), cv.FILLED)
        cv.putText(image, label, (facebox[0], facebox[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def draw_result1(image, faceboxes):
    """Draw the detection result on image"""
    if faceboxes is not None:
        for result in faceboxes:
            facebox = result

            cv.rectangle(image, (facebox[0], facebox[1]),
                         (facebox[2], facebox[3]), (0, 255, 0))



def draw_box(image, faceboxes, box_color=(255, 255, 255), line_width=4):
    """Draw square boxes on image"""
    for facebox in faceboxes:
        cv.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), box_color, line_width)


def main():
    """The main entrance"""
    cap = cv.VideoCapture(VIDEO)
    while True:
        ret, frame = cap.read()
        confidences, faceboxes = get_facebox(frame, threshold=0.5)
        draw_result(frame, confidences, faceboxes)
        lbp_box = get_lbp_facebox(frame)
        draw_box(frame, lbp_box)
        cv.imshow("detections", frame)
        if cv.waitKey(100) != -1:
            break


if __name__ == '__main__':
    main()
