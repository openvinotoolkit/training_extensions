# Copyright (C) 2019 Intel Corporation
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

import numpy as np

import cv2
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from textile.common import from_list, crop_resize_shift_scale

class ImageRetrieval:

    def __init__(self, model_path, model_backend, model, gallery_path, input_size,
                 multiple_images_per_label=False):
        self.impaths, self.gallery_classes, _, self.text_label_to_class_id = from_list(
            gallery_path, multiple_images_per_label)

        self.input_size = input_size

        if model is None:
            if model_backend == 'tf':
                import tensorflow as tf
                from model import keras_applications_mobilenetv2, keras_applications_resnet50

                if args.model == 'resnet50':
                    self.model = keras_applications_resnet50(
                        tf.keras.layers.Input(shape=(args.input_size, args.input_size, 3)))
                if args.model == 'mobilenet_v2':
                    self.model = keras_applications_mobilenetv2(
                        tf.keras.layers.Input(shape=(args.input_size, args.input_size, 3)))

                self.model.load_weights(args.model_weights)
            else:
                from openvino.inference_engine import IENetwork, IEPlugin
                class IEModel():

                    def __init__(self, model_path):
                        self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
                        path = '.'.join(model_path.split('.')[:-1])
                        self.net = IENetwork(model=path + '.xml', weights=path + '.bin')
                        self.plugin.load(network=self.net)
                        self.exec_net = self.plugin.load(network=self.net)

                    def predict(self, image):
                        assert len(image.shape) == 4

                        image = np.transpose(image, (0, 3, 1, 2))
                        out = self.exec_net.infer(inputs={'Placeholder': image})[
                            'model/flatten/Reshape']
                        out = out / np.linalg.norm(out, axis=-1)

                        return out

                self.model = IEModel(model_path)
        else:
            self.model = model

        self.embeddings = self.compute_gallery_embeddings()

    def compute_embedding(self, image):
        image = crop_resize_shift_scale(image, self.input_size)
        embedding = self.model.predict(image)
        return embedding

    def search_in_gallery(self, embedding):
        distances = cosine_distances(embedding, self.embeddings).reshape([-1])
        sorted_indexes = np.argsort(distances)
        return sorted_indexes, distances

    def compute_gallery_embeddings(self):
        images = []

        for full_path in tqdm(self.impaths, desc='Reading gallery images.'):
            image = cv2.imread(full_path)
            if image is None:
                print("ERROR: cannot find image, full_path =", full_path)
            image = crop_resize_shift_scale(image, self.input_size)
            images.append(image)

        embeddings = [None for _ in self.impaths]

        index = 0
        for image in tqdm(images, desc='Computing embeddings of gallery images.'):
            embeddings[index] = self.model.predict(image).reshape([-1])
            index += 1

        return embeddings
