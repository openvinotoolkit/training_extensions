// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ZoomProvider } from '@/components/zoom/zoom.provider';
import { screen } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { render } from 'test-utils/render';

import { MediaCanvas } from './media-canvas';

const mediaItem = getMockedMediaImage({ width: 100, height: 100 });
const image = new ImageData(new Uint8ClampedArray(4 * 100 * 100), 100, 100);

const renderApp = ({ isLoadingOverlay }: { isLoadingOverlay: boolean }) => {
    return render(
        <ZoomProvider>
            <MediaCanvas mediaItem={mediaItem} image={image} isLoadingOverlay={isLoadingOverlay} />
        </ZoomProvider>
    );
};

describe('MediaCanvas', () => {
    it('does not render a loading overlay by default', () => {
        renderApp({ isLoadingOverlay: false });

        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });

    it('renders the loading overlay outside of the zoom transform, so that it is not scaled with the media', () => {
        renderApp({ isLoadingOverlay: true });

        const overlay = screen.getByRole('progressbar');

        expect(overlay).toBeInTheDocument();
        expect(screen.getByTestId('zoom-transform')).not.toContainElement(overlay);
    });
});
