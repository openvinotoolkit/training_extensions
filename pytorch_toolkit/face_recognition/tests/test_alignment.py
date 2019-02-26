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

import unittest
import cv2 as cv
import numpy as np

from utils.face_align import FivePointsAligner
from utils.landmarks_augmentation import RandomRotate


class FaceAlignmentTests(unittest.TestCase):
    """Tests for alignment methods"""
    def test_align_image(self):
        """Synthetic test for alignment function"""
        image = np.zeros((128, 128, 3), dtype=np.float32)
        for point in FivePointsAligner.ref_landmarks:
            point_scaled = point * [128, 128]
            cv.circle(image, tuple(point_scaled.astype(np.int)), 5, (255, 255, 255), cv.FILLED)

        transform = RandomRotate(40., p=1.)
        rotated_data = transform({'img': image, 'landmarks': FivePointsAligner.ref_landmarks})
        aligned_image = FivePointsAligner.align(rotated_data['img'], \
                                                rotated_data['landmarks'].reshape(-1),
                                                d_size=(128, 128), normalized=True)

        for point in FivePointsAligner.ref_landmarks:
            point_scaled = (point * [128, 128]).astype(np.int)
            check_sum = np.mean(aligned_image[point_scaled[1] - 3 : point_scaled[1] + 3,
                                              point_scaled[0] - 3 : point_scaled[0] + 3])
            self.assertGreaterEqual(check_sum, 220)

if __name__ == '__main__':
    unittest.main()
