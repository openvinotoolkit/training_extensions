// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '@/api';
import { createI18nInstance } from '@/i18n';
import { HttpResponse, http as mswHttp } from 'msw';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getVideoFileInitialConfig, prepareVideoFileFormData, videoFileBodyFormatter } from './utils';

vi.mock('../../../../api/tus-upload', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../../../../api/tus-upload')>()),
    transferFile: vi.fn(async () => '00000000-0000-4000-8000-000000000001'),
}));

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
            http.post('/api/sources/media:from-upload', () => {
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

    it('returns a rollback that deletes the uploaded file by its UUID', async () => {
        const sourceMediaId = '712750b2-5a82-47ee-8fba-f3dc96cb615d';
        const deletedIds: string[] = [];
        server.use(
            http.post('/api/sources/media:from-upload', () => {
                return HttpResponse.json(
                    { video_path: `C:\\data\\source_media\\${sourceMediaId}\\sample.mp4` },
                    { status: 201 }
                );
            }),
            http.delete('/api/sources/media/{source_media_id}', ({ params }) => {
                deletedIds.push(params.source_media_id);
                return HttpResponse.json({ deleted_video_path: '' });
            })
        );

        const file = new File(['fake-video-bytes'], 'sample.mp4', { type: 'video/mp4' });
        const formData = buildFormData({ id: '1', name: 'My source', video_path: '', video_file: file, loop: '' });

        const rollback = await prepareVideoFileFormData(formData);
        expect(deletedIds).toEqual([]);

        await rollback?.();

        expect(deletedIds).toEqual([sourceMediaId]);
    });

    it('returns no rollback when no file was uploaded', async () => {
        const formData = buildFormData({ id: '1', name: 'My source', video_path: '/a/b.mp4', loop: '' });

        await expect(prepareVideoFileFormData(formData)).resolves.toBeUndefined();
    });

    it('rejects when the upload fails', async () => {
        server.use(
            mswHttp.post(`${API_BASE_URL}/api/sources/media:from-upload`, () => {
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
