// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';

export const useGetTimmFamilies = (isEnabled: boolean) => {
    return $api.useQuery('get', '/api/model_architectures/timm/families', {}, { enabled: isEnabled });
};

export const useGetTimmVariants = (family: string | null) => {
    return $api.useQuery(
        'get',
        '/api/model_architectures/timm/families/{family}/variants',
        { params: { path: { family: String(family) } } },
        { enabled: family !== null }
    );
};

export const useGetTimmPretrainedTags = (family: string | null, variant: string | null) => {
    return $api.useQuery(
        'get',
        '/api/model_architectures/timm/families/{family}/variants/{variant}/pretrained-tags',
        { params: { path: { family: String(family), variant: String(variant) } } },
        { enabled: family !== null && variant !== null }
    );
};

export const useGetTimmManifest = (family: string | null, variant: string | null, pretrainedTag: string | null) => {
    return $api.useQuery(
        'get',
        '/api/model_architectures/timm/manifest',
        {
            params: {
                query: { family: String(family), variant: String(variant), pretrained_tag: String(pretrainedTag) },
            },
        },
        { enabled: family !== null && variant !== null && pretrainedTag !== null }
    );
};
