import math
import numpy as np

class Metrics(object):
    def __init__(self, name):
        self.name = name
        self.accumulator = 0.0
        self.samples = 0.0

    def update(self, ground, predict):
        self.samples = self.samples + 1

    def get(self):
        return self.accumulator / self.samples

    def reset(self):
        self.accumulator = 0.0
        self.samples = 0.0


def _PSNR(pred, gt, shave_border=0):
    pred = np.asarray(pred).astype(np.float)
    gt = np.asarray(gt).astype(np.float)

    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred - gt) #rgb color space

    r = imdff[:,:,0]
    g = imdff[:,:,1]
    b = imdff[:,:,2]

    y = (r * 65.738 + g * 129.057 + b * 25.064) / 256

    mse = np.mean(y ** 2)
    if mse == 0:
        return np.Infinity

    return - 10 * math.log10(mse)



class PSNR(Metrics):
    def __init__(self, name='PSNR', border=0, input_index=0, target_index=0):
        super(PSNR, self).__init__(name)
        self.border = border
        self.input_index = input_index
        self.target_index = target_index

    def update(self, ground, predict):
        pred = predict[self.input_index].cpu().detach().numpy()
        gr = ground[self.target_index].cpu().detach().numpy()

        assert (gr.shape == pred.shape)

        for i in range(gr.shape[0]):
            _pred = pred[i].T
            _gr = gr[i].T
            psnr = _PSNR(_pred, _gr, self.border)
            if not np.isinf(psnr):
                self.accumulator += psnr
                self.samples += 1


class RMSE(Metrics):
    def __init__(self, name='RMSE', border=4, input_index=0, target_index=0):
        super(RMSE, self).__init__(name)
        self.input_index = input_index
        self.target_index = target_index
        self.border = border

    def update(self, ground, predict):
        h, w = ground[0].shape[2:]
        pred = predict[self.input_index].cpu().detach().numpy()[:,:,self.border:h-self.border,self.border:w-self.border]
        gr = ground[self.target_index].cpu().detach().numpy()[:,:,self.border:h-self.border,self.border:w-self.border]

        assert (gr.shape == pred.shape)

        self.accumulator += ((((pred-gr)**2).sum(axis=(1, 2, 3))/np.prod(gr.shape[1:]))**0.5).mean()

        self.samples += 1