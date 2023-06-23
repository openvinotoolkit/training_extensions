from typing import OrderedDict
import torch
import numpy as np


class ReciproCamHook:
    def __init__(self, model, device, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.device = device
        self.feature = None
        filter = [[1/16.0, 1/8.0, 1/16.0],
                    [1/8.0, 1/4.0, 1/8.0],
                    [1/16.0, 1/8.0, 1/16.0]]
        self.gaussian = torch.tensor(filter).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.target_layers = []
        self.conv_depth = 0
        if self.target_layer_name is not None:
            children = dict(self.model.named_children())
            target = children[self.target_layer_name]
            self.find_target_layer(target)
        else:
            self.find_target_layer(self.model)
        self.target_layers[-1].register_forward_hook(self.cam_hook())


    def find_target_layer(self, m, depth=0):
        children = dict(m.named_children())
        if children == {}:
            if isinstance(m, torch.nn.Conv2d):
                self.target_layers.clear()
                self.target_layers.append(m)
                self.conv_depth = depth
            elif self.conv_depth == depth and len(self.target_layers) > 0 and isinstance(m, torch.nn.BatchNorm2d):
                self.target_layers.append(m)
            elif self.conv_depth == depth and len(self.target_layers) > 0 and isinstance(m, torch.nn.ReLU):
                self.target_layers.append(m)
        else:
            for name, child in children.items():
                self.find_target_layer(child, depth+1)


    def cam_hook(self):
        def fn(_, input, output):
            self.feature = output[0].unsqueeze(0)
            bs, nc, h, w = self.feature.shape
            #print(self.feature.shape)
            new_features = self.mosaic_feature(self.feature, nc, h, w, False)
            new_features = torch.cat((self.feature, new_features), dim = 0)
            return new_features

        return fn


    def mosaic_feature(self, feature_map, nc, h, w, is_gaussian=False):
        new_features = torch.zeros(h*w, nc, h, w).to(self.device)
        if is_gaussian == False:
            for b in range(h*w):
                for i in range(h):
                    for j in range(w):
                        if b == i*w + j:
                            new_features[b,:,i,j] = feature_map[0,:,i,j]
        else:
            for b in range(h*w):
                for i in range(h): #0...h-1
                    kx_s = max(i-1, 0)
                    kx_e = min(i+1, h-1)
                    if i == 0: sx_s = 1
                    else: sx_s = 0
                    if i == h-1: sx_e = 1
                    else: sx_e = 2
                    for j in range(w): #0...w-1
                        ky_s = max(j-1, 0)
                        ky_e = min(j+1, w-1)
                        if j == 0: sy_s = 1
                        else: sy_s = 0
                        if j == w-1: sy_e = 1
                        else: sy_e = 2
                        if b == i*w + j:
                            r_feature_map = feature_map[0,:,i,j].reshape(feature_map.shape[1],1,1)
                            r_feature_map = r_feature_map.repeat(1,3,3)
                            score_map = r_feature_map*self.gaussian.repeat(feature_map.shape[1],1,1)
                            new_features[b,:,kx_s:kx_e+1,ky_s:ky_e+1] = score_map[:,sx_s:sx_e+1,sy_s:sy_e+1]

        return new_features


    def get_class_activaton_map(self, mosaic_predic, index, h, w):
        cam = (mosaic_predic[:,index]).reshape((h, w))
        cam_min = cam.min()
        cam = (cam - cam_min) / (cam.max() - cam_min)
    
        return cam


    def __call__(self, input, index=None):
        with torch.no_grad():
            prediction = self.model(input)
            prediction = self.softmax(prediction)

            if index == None:
                index = prediction[0].argmax().item()

            bs, nc, h, w = self.feature.shape
            cam = self.get_class_activaton_map(prediction[1:, :], index, h, w)

        return cam, index

