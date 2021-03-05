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
import SimpleITK as sitk
import torch


class Metrics():
    def __init__(self, name):
        self.name = name
        self.accumulator = 0.0
        self.samples = 0.0

    #pylint: disable=W0613
    def update(self, ground, predict):
        self.samples = self.samples + 1

    def get(self):
        return self.accumulator / self.samples

    def reset(self):
        self.accumulator = 0.0
        self.samples = 0.0

class Dice(Metrics):
    def __init__(self, name='Dice', input_index=0, target_index=0, classes=5):
        super(Dice, self).__init__(name)
        self.input_index = input_index
        self.target_index = target_index
        self.classes = classes

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()
        gr = ground[self.target_index].detach()

        assert gr.shape == pred.shape

        pred = ((torch.argmax(pred, dim=1) + 1) * (torch.max(pred, dim=1)[0] > 0.5).long()).long()
        gr = ((torch.argmax(gr, dim=1) + 1) * (torch.max(gr, dim=1)[0] > 0.5).long()).long()

        result = np.zeros(shape=(pred.shape[0], self.classes-1))

        for i in range(1, self.classes):
            p = (pred == i).float()
            g = (gr == i).float()
            #print(p.max(), g.max())
            r = 2 * (p * g).sum(dim=(1, 2, 3))/((p+g).sum(dim=(1, 2, 3))+1e-6)
            #print(r.shape)
            result[:, i-1] = r.cpu().numpy()

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1


class Hausdorff_ITK(Metrics):
    def __init__(self, name='Hausdorff_ITK', input_index=0, target_index=0, classes=5):
        super(Hausdorff_ITK, self).__init__(name)
        self.input_index = input_index
        self.target_index = target_index
        self.classes = classes
        self.hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()
        gr = ground[self.target_index].detach()

        assert gr.shape == pred.shape

        pred = ((torch.argmax(pred, dim=1) + 1) * (torch.max(pred, dim=1)[0] > 0.5).long()).long().cpu().numpy()
        gr = ((torch.argmax(gr, dim=1) + 1) * (torch.max(gr, dim=1)[0] > 0.5).long()).long().cpu().numpy()

        result = np.zeros(shape=(pred.shape[0], self.classes-1))

        for n in range(pred.shape[0]):
            for i in range(1, self.classes):
                p = (pred[n] == i).astype(np.uint8)
                g = (gr[n] == i).astype(np.uint8)


                r = 1e+6
                try:
                    self.hausdorff_distance_filter.Execute(sitk.GetImageFromArray(g), sitk.GetImageFromArray(p))
                    r = self.hausdorff_distance_filter.GetHausdorffDistance()
                except RuntimeError:
                    print("Hausdorff_ITK:RuntimeError")

                result[n, i-1] = r

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1


def print_metrics(writer, metric, prefix, epoch):
    if isinstance(metric.get(), np.ndarray):
        for i in range(metric.get().shape[0]):
            writer.add_scalar(prefix + metric.name+str(i), metric.get()[i], epoch)
    else:
        writer.add_scalar(prefix + metric.name, metric.get(), epoch)

    print('Epoch %d, %s %s %s' % (epoch, prefix, metric.name, metric.get()))
