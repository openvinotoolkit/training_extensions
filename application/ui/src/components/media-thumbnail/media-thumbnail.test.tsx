// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { useIsScrolling } from '../../hooks/use-is-scrolling.hook';
import { MediaThumbnail } from './media-thumbnail.component';

vi.mock('../../hooks/use-is-scrolling.hook', () => ({
    useIsScrolling: vi.fn(),
}));

const getImage = () => screen.getByRole('img', { name: 'Test Image' });
const getSkeleton = () => screen.getByRole('img', { name: 'Loading…' });
const querySkeleton = () => screen.queryByRole('img', { name: 'Loading…' });

describe('MediaThumbnail', () => {
    beforeEach(() => {
        vi.mocked(useIsScrolling).mockReturnValue(false);
    });

    it('calls onClick when image is clicked', async () => {
        const mockedClick = vi.fn();
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' onClick={mockedClick} item={{ type: 'image' }} />);

        await userEvent.click(getImage());

        expect(mockedClick).toHaveBeenCalled();
    });

    it('calls onDoubleClick when image is double-clicked', async () => {
        const mockedDblClick = vi.fn();
        render(
            <MediaThumbnail
                url='test-image.jpg'
                alt='Test Image'
                onDoubleClick={mockedDblClick}
                item={{ type: 'image' }}
            />
        );

        await userEvent.dblClick(getImage());

        expect(mockedDblClick).toHaveBeenCalled();
    });

    it('displays frames count when item is a video', () => {
        render(
            <MediaThumbnail
                url='test-video.mp4'
                alt='Test Image'
                item={{ type: 'video', frame_count: 3600, annotated_frame_count: 10, duration: 60 }}
            />
        );

        expect(screen.getByText('01:00')).toBeInTheDocument();
    });

    it('does not set the image src while scrolling', () => {
        vi.mocked(useIsScrolling).mockReturnValue(true);

        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' item={{ type: 'image' }} />);

        expect(getImage()).not.toHaveAttribute('src');
    });

    it('sets the image src when not scrolling', () => {
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' item={{ type: 'image' }} />);

        expect(getImage()).toHaveAttribute('src', 'test-image.jpg');
    });

    it('shows the skeleton again when the url changes', async () => {
        const Thumbnail = () => {
            const [url, setUrl] = useState('test-image.jpg');

            return (
                <>
                    <button onClick={() => setUrl('other-image.jpg')}>Next</button>
                    <MediaThumbnail url={url} alt='Test Image' item={{ type: 'image' }} />
                </>
            );
        };

        render(<Thumbnail />);
        fireEvent.load(getImage());

        await userEvent.click(screen.getByRole('button'));

        // CSS modules keep the local name, so assert on the class that hides the image.
        expect(getImage().className).toContain('imgHidden');
        expect(getSkeleton()).toBeInTheDocument();
    });

    it('stops the skeleton but keeps the image hidden when the thumbnail fails to load', () => {
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' item={{ type: 'image' }} />);

        // jsdom never loads the image, so mark it as settled the way a broken image is.
        Object.defineProperty(getImage(), 'complete', { value: true });
        fireEvent.error(getImage());

        expect(getImage().className).toContain('imgHidden');
        expect(querySkeleton()).not.toBeInTheDocument();
    });

    it('ignores the error fired while the image is still loading', () => {
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' item={{ type: 'image' }} />);

        expect(getImage()).toHaveProperty('complete', false);

        fireEvent.error(getImage());

        expect(getSkeleton()).toBeInTheDocument();
    });
});
