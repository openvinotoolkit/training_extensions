import torch
import numpy as np


class ReciproCam:
    def __init__(self, feature_net, head_net, device):
        self.feature_net = feature_net
        self.head_net = head_net
        self.feature_net.eval()
        self.head_net.eval()
        self.device = device
        self.feature = None
        self.softmax = torch.nn.Softmax(dim=1)
        filter = [[1/16.0, 1/8.0, 1/16.0],
                    [1/8.0, 1/4.0, 1/8.0],
                    [1/16.0, 1/8.0, 1/16.0]]
        self.gaussian = torch.tensor(filter).to(device)


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

        if type(index) == list:
            indexes = torch.as_tensor(index).to(self.device)
            cam = torch.index_select(mosaic_predic, 1, indexes)
            cam = cam.sum(dim=1)
            cam = cam.reshape((h, w))
        else:
            cam = (mosaic_predic[:,index]).reshape((h, w))
        cam_min = cam.min()
        cam = (cam - cam_min) / (cam.max() - cam_min)
    
        return cam


    def get_feature_map(self):
        act_map = None
        if self.feature is not None:
            act_map = self.feature.mean(dim=0)
            act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())

        return act_map

    def __call__(self, input, index=None):

        with torch.no_grad():
            feature = self.feature_net(input)
            prediction = self.head_net(feature)
            self.feature = feature.squeeze(0)

            if index == None:
                index = np.argmax(prediction.cpu().data.numpy())

            bs, nc, h, w = feature.shape

            new_features = self.mosaic_feature(feature, nc, h, w, True)
            new_prediction = self.head_net(new_features)
            new_s_prediction = self.softmax(new_prediction)
            cam = self.get_class_activaton_map(new_s_prediction, index, h, w)

        return cam, index

