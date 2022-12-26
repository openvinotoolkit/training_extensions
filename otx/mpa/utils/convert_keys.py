# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from otx.mpa.utils.logger import get_logger

logger = get_logger()


def convert_keys(name, path, new_path=None):
    if not path:
        return path

    if not new_path:
        new_path = path[:-3] + "converted.pth"
    if torch.cuda.is_available():
        ckpt = torch.load(path)
    else:
        ckpt = torch.load(path, map_location="cpu")

    new_ckpt = {}

    if "state_dict" in ckpt.keys():
        state_dict = ckpt.pop("state_dict")
        new_ckpt["state_dict"] = {}
        new_state_dict = new_ckpt["state_dict"]
        for k, v in ckpt.items():
            new_ckpt[k] = v
    else:
        state_dict = ckpt
        new_state_dict = new_ckpt

    if name == "OTEMobileNetV3":
        for k, v in state_dict.items():
            if k.startswith("classifier."):
                new_state_dict["head." + k] = v
            elif not k.startswith("backbone.") and not k.startswith("head."):
                new_state_dict["backbone." + k] = v
            else:
                new_state_dict[k] = v
    elif name == "OTEEfficientNet":
        for k, v in state_dict.items():
            # if 'output.' in k:
            #     new_state_dict['head.'+k[7:]] = v
            # else:
            #     new_state_dict['backbone.'+k] = v
            if k.startswith("output."):
                v = v.t()
                if "asl" in k:
                    new_state_dict["head.fc" + k[10:]] = v
                else:
                    new_state_dict["head." + k[7:]] = v
            elif not k.startswith("backbone.") and not k.startswith("head."):
                if "activ." in k:
                    k = k.replace("activ.", "activate.")
                new_state_dict["backbone." + k] = v
            else:
                new_state_dict[k] = v
    else:
        raise ValueError(f"Not supported model - {name}")

    if new_state_dict == state_dict:
        logger.info("conversion is not required.")
        return path

    torch.save(new_ckpt, new_path)
    return new_path
