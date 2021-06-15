import torch

def get_shape(x):
    return x.shape
    #this is to get more compact and human readable but unresizable onnx model
    #return [int(s) for s in x.shape]

#class to simplify work with states
class States():
    def __init__(self, state_old, state=None):
        self.state_old =  None if state_old is None else state_old.copy()
        self.state = [] if state is None else state.copy()

    def update(self, state):
        if state is not None:
            self.state += [s.detach() for s in state]
        if self.state_old is not None:
            self.state_old = self.state_old[len(state):]

    def pad_left(self, x, size, dim, shift_right=False):
        if self.state_old is None:
            shape = get_shape(x)
            shape[dim] = size
            x_pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
        else:
            x_pad = self.state_old[0]

        #add left part of x
        x_padded = torch.cat([x_pad, x], dim)

        #get right part for padding on next iter
        x_splited = torch.split(x_padded, [get_shape(x_padded)[dim]-size, size], dim=dim)

        self.update(x_splited[-1:])

        if shift_right:
            return x_splited[0]
        else:
            return x_padded

