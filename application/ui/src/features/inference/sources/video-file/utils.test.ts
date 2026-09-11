// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from '@/i18n';
import { HttpResponse } from 'msw';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getVideoFileInitialConfig, prepareVideoFileFormData, videoFileBodyFormatter } from './utils';

const buildFormData = (fields: Record<string, string | Blob>): FormData => {
    const formData = new FormData();

    Object.entries(fields).forEach(([key, value]) => {
        formData.append(key, value);
    });

    return formData;
};

describe('getVideoFileInitialConfig', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    it('returns a config with an empty video_path and a unique name', () => {
        expect(getVideoFileInitialConfig(t, ['Video file source'])).toEqual({
            id: '',
            name: 'Video file source (1)',
            source_type: 'video_file',
            video_path: '',
            loop: false,
        });
    });
});

describe('videoFileBodyFormatter', () => {
    it('reads video_path as-is from the FormData', () => {
        const formData = buildFormData({
            id: '1',
            name: 'My source',
            video_path: '/a/b.mp4',
            loop: 'on',
        });

        expect(videoFileBodyFormatter(formData)).toEqual({
            id: '1',
            name: 'My source',
            source_type: 'video_file',
            video_path: '/a/b.mp4',
            loop: true,
        });
    });
});

describe('prepareVideoFileFormData', () => {
    it('does nothing when no file was selected, leaving the typed video_path untouched', async () => {
        const formData = buildFormData({
            id: '1',
            name: 'My source',
            video_path: '/a/b.mp4',
            loop: '',
        });

        await prepareVideoFileFormData(formData);

        expect(formData.get('video_path')).toBe('/a/b.mp4');
    });

    it('uploads the selected file and overwrites video_path with the returned path', async () => {
        const resolvedPath = '/data/source_media/uuid/sample.mp4';
        server.use(
            http.post('/api/sources/media', () => {
                return HttpResponse.json({ video_path: resolvedPath }, { status: 201 });
            })
        );

        const file = new File(['fake-video-bytes'], 'sample.mp4', { type: 'video/mp4' });
        const formData = buildFormData({
            id: '1',
            name: 'My source',
            video_path: '',
            video_file: file,
            loop: '',
        });

        await prepareVideoFileFormData(formData);

        expect(formData.get('video_path')).toBe(resolvedPath);
    });

    it('rejects when the upload fails', async () => {
        server.use(
            http.post('/api/sources/media', () => {
                // The 422 response has no documented schema in the OpenAPI spec (description only).
                // @ts-expect-error There is an incorrect type in OpenAPI
                return HttpResponse.json({ detail: 'Unsupported video format' }, { status: 422 });
            })
        );

        const file = new File(['fake-video-bytes'], 'sample.mp4', { type: 'video/mp4' });
        const formData = buildFormData({
            id: '1',
            name: 'My source',
            video_path: '',
            video_file: file,
            loop: '',
        });

        await expect(prepareVideoFileFormData(formData)).rejects.toBeTruthy();
    });
});
