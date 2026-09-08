// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { uploadSourceVideo } from '@/api';
import type { VideoFileSourceConfig } from '@/api/types';

import { getUniqueName } from '../utils';

export const getVideoFileInitialConfig = (existingNames: string[] = []): VideoFileSourceConfig => ({
    id: '',
    name: getUniqueName('Video file source', existingNames),
    source_type: 'video_file',
    video_path: '',
    loop: false,
});

// Uploads the selected file (if any) and writes the resulting path back into `video_path`, so
// `videoFileBodyFormatter` can stay a plain, synchronous formatter like its sibling sources.
export const prepareVideoFileFormData = async (formData: FormData): Promise<void> => {
    const file = formData.get('video_file');

    // An untouched file input still yields a File entry (empty filename) once it has a `name`,
    // so only treat it as "a file was selected" when it actually has a name.
    if (file instanceof File && file.name !== '') {
        const { video_path } = await uploadSourceVideo(file);

        formData.set('video_path', video_path);
    }
};

export const videoFileBodyFormatter = (formData: FormData): VideoFileSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'video_file',
    video_path: String(formData.get('video_path')),
    loop: formData.get('loop') === 'on' ? true : false,
});
