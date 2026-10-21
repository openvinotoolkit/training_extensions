// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getDatasetRevisionThumbnailUrl,
    getMediaBinaryUrl,
    getMediaDownloadUrl,
    getProjectThumbnailUrl,
    getThumbnailUrl,
    getVideoFrameBinaryUrl,
    getVideoFrameThumbnailUrl,
} from './media-url.utils';

const mocks = rs.hoisted(() => ({ apiBaseUrl: '' }));

rs.mock('../api/client', () => ({
    get API_BASE_URL() {
        return mocks.apiBaseUrl;
    },
}));

describe.each([
    { name: 'absolute base (dev/preview)', base: 'http://localhost:7860', prefix: 'http://localhost:7860' },
    { name: 'absolute base with trailing slash', base: 'http://localhost:7860/', prefix: 'http://localhost:7860' },
    { name: 'empty base (Docker, same origin)', base: '', prefix: '' },
])('media-url helpers with $name', ({ base, prefix }) => {
    beforeEach(() => {
        mocks.apiBaseUrl = base;
    });

    it('builds the expected URLs', () => {
        expect(getProjectThumbnailUrl('p1')).toBe(`${prefix}/api/projects/p1/thumbnail`);
        expect(getThumbnailUrl('p1', 'm1')).toBe(`${prefix}/api/projects/p1/dataset/media/m1/thumbnail`);
        expect(getMediaBinaryUrl('p1', 'm1')).toBe(`${prefix}/api/projects/p1/dataset/media/m1/binary`);
        expect(getMediaDownloadUrl('p1', 'm1')).toBe(`${prefix}/api/projects/p1/dataset/media/m1/binary?raw=true`);
        expect(getDatasetRevisionThumbnailUrl('p1', 'r1', 'm1')).toBe(
            `${prefix}/api/projects/p1/dataset_revisions/r1/items/m1/thumbnail`
        );
        expect(getVideoFrameBinaryUrl('p1', 'm1', 7)).toBe(
            `${prefix}/api/projects/p1/dataset/media/m1/binary?frame_index=7`
        );
        expect(getVideoFrameThumbnailUrl('p1', 'm1', 7)).toBe(
            `${prefix}/api/projects/p1/dataset/media/m1/thumbnail?frame_index=7`
        );
    });
});
