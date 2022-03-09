# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch
import multiprocessing as mp



def maximum_path_np(map, mask):
    """
    value: [b, t_text, t_mel]
    mask: [b, t_text, t_mel]
    """
    map = map * mask

    device = map.device
    dtype = map.dtype

    map = map.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(map).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_text_max = mask.sum(1)[:, 0].astype(np.int32)
    t_mel_max = mask.sum(2)[:, 0].astype(np.int32)


    for b in range(map.shape[0]):
        min_val = -1e9
        t_text, t_mel = t_text_max[b], t_mel_max[b]
        index = t_text - 1

        for x in range(t_mel):
            for y in range(max(0, t_text + x - t_mel), min(t_text, x + 1)):
                if x == y:
                    v_cur = min_val
                else:
                    v_cur = map[b, y, x - 1]
                if y == 0:
                    if x == 0:
                        v_prev = 0.
                    else:
                        v_prev = min_val
                else:
                    v_prev = map[b, y - 1, x - 1]
                map[b, y, x] = max(v_cur, v_prev) + map[b, y, x]

        for x in range(t_mel - 1, -1, -1):
            path[b, index, x] = 1
            if index != 0 and (index == x or map[b, index, x - 1] < map[b, index - 1, x - 1]):
                index = index - 1


    return torch.from_numpy(path).to(device=device, dtype=dtype)
