# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from nncf.torch.nncf_network import NNCFNetwork

from otx.algorithms.common.adapters.nncf.patches import nncf_train_step

# add wrapper train_step method
NNCFNetwork.train_step = nncf_train_step
