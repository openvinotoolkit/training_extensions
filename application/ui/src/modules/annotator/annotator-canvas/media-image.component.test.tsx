// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedMediaImage, getMockedVideo, getMockedVideoFrame } from 'mocks/mock-media';
import { render } from 'test-utils/render';

import { MediaImage } from './media-image.component';

rs.mock('../video-player/video-frame.component', () => ({
    VideoFrame: () => <div data-testid='video-frame' />,
}));

const makeImageData = (width: number, height: number): ImageData =>
    ({ width, height, data: new Uint8ClampedArray(width * height * 4) }) as ImageData;

describe('MediaImage', () => {
    it('renders a canvas for a full-resolution image', () => {
        const mediaItem = getMockedMediaImage({ width: 100, height: 100 });
        const image = makeImageData(100, 100);

        const { container } = render(<MediaImage image={image} mediaItem={mediaItem} />);

        expect(container.querySelector('canvas')).toBeInTheDocument();
        expect(container.querySelector('img')).not.toBeInTheDocument();
    });

    it('renders an img for a downscaled (non-full-resolution) image', () => {
        const mediaItem = getMockedMediaImage({ width: 200, height: 200 });
        // Image decoded at half resolution — not full-res.
        const image = makeImageData(100, 100);

        const { container } = render(<MediaImage image={image} mediaItem={mediaItem} />);

        expect(container.querySelector('img')).toBeInTheDocument();
        expect(container.querySelector('canvas')).not.toBeInTheDocument();
    });

    it('renders a canvas with a video-frame overlay for a video-frame media item', () => {
        const mediaItem = getMockedVideoFrame({ width: 400, height: 400 });
        const image = makeImageData(400, 400);

        const { container } = render(<MediaImage image={image} mediaItem={mediaItem} />);

        expect(container.querySelector('canvas')).toBeInTheDocument();
        expect(screen.getByTestId('video-frame')).toBeInTheDocument();
        expect(container.querySelector('img')).not.toBeInTheDocument();
    });

    it('renders a canvas with a video-frame overlay for a video media item', () => {
        const mediaItem = getMockedVideo({ width: 400, height: 400 });
        const image = makeImageData(400, 400);

        const { container } = render(<MediaImage image={image} mediaItem={mediaItem} />);

        expect(container.querySelector('canvas')).toBeInTheDocument();
        expect(screen.getByTestId('video-frame')).toBeInTheDocument();
        expect(container.querySelector('img')).not.toBeInTheDocument();
    });
});
