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

import torchreid
from .engine import ImageAMSoftmaxEngine


def build_engine(cfg, datamanager, model, optimizer, scheduler, writer=None, openvino_model=None):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = ImageAMSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                cfg.reg,
                cfg.metric_losses,
                cfg.data.transforms.batch_transform,
                scheduler,
                cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                conf_penalty=cfg.loss.softmax.conf_pen,
                softmax_type='stock',
                writer=writer,
                openvino_model=openvino_model,
            )
        elif cfg.loss.name == 'am_softmax':
            engine = ImageAMSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                cfg.reg,
                cfg.metric_losses,
                cfg.data.transforms.batch_transform,
                scheduler,
                cfg.use_gpu,
                conf_penalty=cfg.loss.softmax.conf_pen,
                softmax_type='am',
                m=cfg.loss.softmax.m,
                s=cfg.loss.softmax.s,
                writer=writer,
                openvino_model=openvino_model,
            )
        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine
