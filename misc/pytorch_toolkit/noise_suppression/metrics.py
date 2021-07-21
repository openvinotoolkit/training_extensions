"""
 Copyright (c) 2021 Intel Corporation

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

import torch

EPS = torch.finfo(torch.float32).tiny

def sisdr(y,
          target,
          db_flag = True,
          reg_pow_db = None,
          inv_flag=False,
          mean_flag=False,
          onelog_flag=False,
          beta_flag=False):

    dim_agregate = -1

    reg_pow = EPS if reg_pow_db is None else 10**(reg_pow_db/10) * y.shape[1]

    target = target - torch.mean(target, dim=dim_agregate, keepdim=True)
    y = y - torch.mean(y, dim=dim_agregate, keepdim=True)


    acc_f = torch.mean if mean_flag else torch.sum

    y_by_target = acc_f(y * target, dim=dim_agregate, keepdim=True)
    t2 = acc_f(target ** 2, dim=dim_agregate, keepdim=True)

    if beta_flag:
        #scale model output to target
        y_target = target
        beta = (t2 + reg_pow)/(y_by_target+reg_pow)
        y_noise = y * beta - target
    else:
        #scale target to model output
        alfa = y_by_target/(t2 + reg_pow)
        y_target = alfa * target
        y_noise = y - y_target

    target_pow = acc_f(y_target ** 2, dim=dim_agregate)
    noise_pow = acc_f(y_noise ** 2, dim=dim_agregate)

    if db_flag:
        if onelog_flag:
            l = -10 * torch.log10((noise_pow + reg_pow)/(target_pow + reg_pow))
        else:
            l = 10 * torch.log10(target_pow + reg_pow) - 10 * torch.log10(noise_pow + reg_pow)
        if inv_flag:
            l = -l
    else:
        if inv_flag:
            l = (noise_pow + reg_pow) / (target_pow + reg_pow)
        else:
            l = (target_pow + reg_pow) / (noise_pow + reg_pow)
    return l
