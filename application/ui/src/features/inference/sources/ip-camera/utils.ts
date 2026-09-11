// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { IPCameraSourceConfig } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { getUniqueName } from '../utils';

export const getIpCameraInitialConfig = (t: TranslateFn, existingNames: string[] = []): IPCameraSourceConfig => ({
    id: '',
    name: getUniqueName(t('inference.sources.defaultNames.ipCamera'), existingNames),
    source_type: 'ip_camera',
    stream_url: '',
    auth_required: false,
});

export const ipCameraBodyFormatter = (formData: FormData): IPCameraSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'ip_camera',
    stream_url: String(formData.get('stream_url')),
    auth_required: String(formData.get('auth_required')) === 'on' ? true : false,
});
