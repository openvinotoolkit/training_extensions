// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';
import * as UTIF from 'utif';

import { http } from '../api/utils';
import { server } from '../msw-node-setup';
import { getImageDataFromTiffUrl, isTiffFormat } from './media-utils';

rs.mock('utif', () => ({
    decode: rs.fn(),
    decodeImage: rs.fn(),
    toRGBA8: rs.fn(),
}));

afterEach(() => {
    rs.resetAllMocks();
    rs.restoreAllMocks();
});

const TIFF_BINARY_URL = '/api/projects/test-project/dataset/media/test-media/binary';

describe('isTiffFormat', () => {
    it.each(['tif', 'tiff', 'TIF', 'TIFF'])('returns true for format "%s"', (format) => {
        expect(isTiffFormat({ format })).toBe(true);
    });

    it.each(['jpg', 'jpeg', 'png', 'webp', 'bmp'])('returns false for format "%s"', (format) => {
        expect(isTiffFormat({ format })).toBe(false);
    });

    it('returns false when format is undefined', () => {
        expect(isTiffFormat({})).toBe(false);
    });
});

describe('getImageDataFromTiffUrl', () => {
    it('returns decoded ImageData for a valid TIFF', async () => {
        const fakeBuffer = new ArrayBuffer(8);
        const fakeRgba = new Uint8Array([255, 0, 0, 255]);
        const fakeTiff = { width: 1, height: 1, data: new Uint8Array(fakeBuffer) };

        server.use(
            http.get(
                '/api/projects/{project_id}/dataset/media/{media_id}/binary',
                () => new HttpResponse(fakeBuffer, { headers: { 'Content-Type': 'image/tiff' } })
            )
        );
        rs.mocked(UTIF.decode).mockReturnValue([fakeTiff]);
        rs.mocked(UTIF.toRGBA8).mockReturnValue(fakeRgba);

        const result = await getImageDataFromTiffUrl(TIFF_BINARY_URL);

        expect(UTIF.decode).toHaveBeenCalledWith(expect.any(ArrayBuffer));
        expect(UTIF.decodeImage).toHaveBeenCalledWith(expect.any(ArrayBuffer), fakeTiff);
        expect(result).toBeInstanceOf(ImageData);
        expect(result.width).toBe(1);
        expect(result.height).toBe(1);
    });

    it('returns 1x1 fallback when decode returns no IFDs', async () => {
        server.use(
            http.get(
                '/api/projects/{project_id}/dataset/media/{media_id}/binary',
                () => new HttpResponse(new ArrayBuffer(0))
            )
        );
        rs.mocked(UTIF.decode).mockReturnValue([]);

        const result = await getImageDataFromTiffUrl(TIFF_BINARY_URL);

        expect(result.width).toBe(1);
        expect(result.height).toBe(1);
    });

    it('returns 1x1 fallback and logs error when fetch fails', async () => {
        const consoleSpy = rs.spyOn(console, 'error').mockImplementation(() => undefined);
        server.use(
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', () => {
                throw new Error('network error');
            })
        );

        const result = await getImageDataFromTiffUrl(TIFF_BINARY_URL);

        expect(result.width).toBe(1);
        expect(result.height).toBe(1);
        expect(consoleSpy).toHaveBeenCalledWith('Failed to decode TIFF image:', expect.any(Error));
    });
});
