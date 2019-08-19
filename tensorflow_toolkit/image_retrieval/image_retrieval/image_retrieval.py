"""
 Copyright (c) 2019 Intel Corporation

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

from tqdm import tqdm

import numpy as np

import cv2
from sklearn.metrics.pairwise import cosine_distances

from image_retrieval.common import from_list, preproces_image

def nothing(image):
    return image


class ImageRetrieval:

    def __init__(self, model_path, model_backend, model, gallery_path, input_size, cpu_extensions,
                 multiple_images_per_label=False):
        self.impaths, self.gallery_classes, _, self.text_label_to_class_id = from_list(
            gallery_path, multiple_images_per_label)

        self.input_size = input_size

        self.preprocess = preproces_image

        if model is None or isinstance(model, str):
            if model_backend == 'tf':
                import tensorflow as tf
                from image_retrieval.model import keras_applications_mobilenetv2, \
                    keras_applications_resnet50

                if model == 'resnet50':
                    self.model = keras_applications_resnet50(
                        tf.keras.layers.Input(shape=(input_size, input_size, 3)))
                if model == 'mobilenet_v2':
                    self.model = keras_applications_mobilenetv2(
                        tf.keras.layers.Input(shape=(input_size, input_size, 3)))

                self.model.load_weights(model_path)
            else:
                from openvino.inference_engine import IENetwork, IECore
                class IEModel():

                    def __init__(self, model_path):
                        ie = IECore()
                        if cpu_extensions:
                            ie.add_extension(cpu_extensions, 'CPU')

                        path = '.'.join(model_path.split('.')[:-1])
                        self.net = IENetwork(model=path + '.xml', weights=path + '.bin')
                        self.exec_net = ie.load_network(network=self.net, device_name='CPU')

                    def predict(self, image):
                        assert len(image.shape) == 4

                        image = np.transpose(image, (0, 3, 1, 2))
                        out = self.exec_net.infer(inputs={'Placeholder': image})[
                            'model/tf_op_layer_mul/mul/Normalize']

                        return out

                self.model = IEModel(model_path)
                self.preprocess = nothing
        else:
            self.model = model

        self.embeddings = self.compute_gallery_embeddings()

    def compute_embedding(self, image):
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.preprocess(image)
        image = np.expand_dims(image, axis=0)
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
            image = cv2.resize(image, (self.input_size, self.input_size))
            image = self.preprocess(image)
            image = np.expand_dims(image, axis=0)
            images.append(image)

        embeddings = [None for _ in self.impaths]

        index = 0
        for image in tqdm(images, desc='Computing embeddings of gallery images.'):
            embeddings[index] = self.model.predict(image).reshape([-1])
            index += 1

        return embeddings
