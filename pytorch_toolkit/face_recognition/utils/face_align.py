"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2 as cv
import numpy as np


class FivePointsAligner():
    """This class performs face alignmet by five reference points"""
    ref_landmarks = np.array([30.2946 / 96, 51.6963 / 112,
                              65.5318 / 96, 51.5014 / 112,
                              48.0252 / 96, 71.7366 / 112,
                              33.5493 / 96, 92.3655 / 112,
                              62.7299 / 96, 92.2041 / 112], dtype=np.float64).reshape(5, 2)
    @staticmethod
    def align(img, landmarks, d_size=(400, 400), normalized=False, show=False):
        """Transforms given image in such a way that landmarks are located near ref_landmarks after transformation"""
        assert len(landmarks) == 10
        assert isinstance(img, np.ndarray)
        landmarks = np.array(landmarks).reshape(5, 2)
        dw, dh = d_size

        keypoints = landmarks.copy().astype(np.float64)
        if normalized:
            keypoints[:, 0] *= img.shape[1]
            keypoints[:, 1] *= img.shape[0]

        keypoints_ref = np.zeros((5, 2), dtype=np.float64)
        keypoints_ref[:, 0] = FivePointsAligner.ref_landmarks[:, 0] * dw
        keypoints_ref[:, 1] = FivePointsAligner.ref_landmarks[:, 1] * dh

        transform_matrix = transformation_from_points(keypoints_ref, keypoints)
        output_im = cv.warpAffine(img, transform_matrix, d_size, flags=cv.WARP_INVERSE_MAP)

        if show:
            tmp_output = output_im.copy()
            for point in keypoints_ref:
                cv.circle(tmp_output, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            for point in keypoints:
                cv.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            img = cv.resize(img, d_size)
            cv.imshow('source/warped', np.hstack((img, tmp_output)))
            cv.waitKey()

        return output_im


def transformation_from_points(points1, points2):
    """Builds an affine transformation matrix form points1 to points2"""
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    u, _, vt = np.linalg.svd(np.matmul(points1.T, points2))
    r = np.matmul(u, vt).T

    return np.hstack(((s2 / s1) * r, (c2.T - (s2 / s1) * np.matmul(r, c1.T)).reshape(2, -1)))
