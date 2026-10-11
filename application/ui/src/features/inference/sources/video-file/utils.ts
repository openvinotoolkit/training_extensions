// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { deleteSourceVideo, uploadSourceVideo } from '@/api';
import type { VideoFileSourceConfig } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { getUniqueName } from '../utils';

export const getVideoFileInitialConfig = (t: TranslateFn, existingNames: string[] = []): VideoFileSourceConfig => ({
    id: '',
    name: getUniqueName(t('inference.sources.defaultNames.videoFile'), existingNames),
    source_type: 'video_file',
    video_path: '',
    loop: false,
});

export type VideoFileUploadOptions = {
    onProgress?: (bytesSent: number, bytesTotal: number) => void;
    signal?: AbortSignal;
};

// Uploads the selected file (if any) and writes the resulting path back into `video_path`, so
// `videoFileBodyFormatter` can stay a plain, synchronous formatter like its sibling sources.
// The returned rollback deletes the upload so it is not left orphaned when the source is not saved.
export const prepareVideoFileFormData = async (
    formData: FormData,
    { onProgress, signal }: VideoFileUploadOptions = {}
): Promise<(() => Promise<void>) | undefined> => {
    const file = formData.get('video_file');

    // An untouched file input still yields a File entry (empty filename) once it has a `name`,
    // so only treat it as "a file was selected" when it actually has a name.
    if (!(file instanceof File) || file.name === '') {
        return undefined;
    }

    const { video_path } = await uploadSourceVideo(file, {
        signal,
        onProgress: (bytesSent) => onProgress?.(bytesSent, file.size),
    });

    formData.set('video_path', video_path);
    // Uploads are stored as `<source_media_dir>/<uuid>/<filename>`; the UUID identifies the upload.
    const sourceMediaId = video_path.split(/[\\/]/).at(-2);

    return sourceMediaId ? () => deleteSourceVideo(sourceMediaId) : undefined;
};

export const videoFileBodyFormatter = (formData: FormData): VideoFileSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'video_file',
    video_path: String(formData.get('video_path')),
    loop: formData.get('loop') === 'on' ? true : false,
});
