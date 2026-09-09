// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { DatasetSubset, MediaImage } from '@/api/types';
import { ZoomProvider } from '@/components/zoom/zoom.provider';
import { type Key } from '@geti-ui/ui';
import { fireEvent, screen } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { render } from 'test-utils/render';

import { AnnotationVisibilityProvider } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { CanvasSettingsProvider } from '../primary-toolbar/settings/canvas-settings-provider.component';
import { BottomToolbar } from './bottom-toolbar.component';

const mockMediaItem = getMockedMediaImage({
    id: 'media-123',
    name: 'test-image',
    format: 'jpg',
    width: 1920,
    height: 1080,
});

type RenderProps = {
    mediaItem?: MediaImage;
    isUserReviewed?: boolean;
    subset?: DatasetSubset;
    isReadOnlySubset?: boolean;
};

const renderBottomToolbar = ({
    mediaItem = mockMediaItem,
    isUserReviewed = false,
    subset = 'unassigned',
    isReadOnlySubset = false,
}: RenderProps = {}) => {
    return render(
        <ZoomProvider>
            <AnnotationVisibilityProvider>
                <CanvasSettingsProvider>
                    <BottomToolbar
                        isReadOnlySubset={isReadOnlySubset}
                        mediaItem={mediaItem}
                        isUserReviewed={isUserReviewed}
                        subset={subset}
                        onSubsetChange={vi.fn()}
                    />
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );
};

describe('BottomToolbar', () => {
    it('displays the filename with correct format and dimensions', () => {
        renderBottomToolbar();

        expect(screen.getByText('test-image.jpg (1920 x 1080 px)')).toBeInTheDocument();
    });

    it('displays "Reviewed" tag when user has reviewed the media', () => {
        renderBottomToolbar({ isUserReviewed: true });

        expect(screen.getByLabelText('Reviewed')).toBeInTheDocument();
    });

    it('displays "For Review" tag when user has not reviewed the media', () => {
        renderBottomToolbar({ isUserReviewed: false });

        expect(screen.getByLabelText('For Review')).toBeInTheDocument();
    });

    it('renders the subset picker with default placeholder', () => {
        renderBottomToolbar();

        expect(screen.getByLabelText('Select subset')).toBeInTheDocument();
    });

    it('renders the subset instead of picker when is read only mode', () => {
        renderBottomToolbar({ subset: 'validation', isReadOnlySubset: true });

        expect(screen.getByLabelText('Validation')).toBeInTheDocument();
    });

    it('shows pending subset tag after selection', () => {
        const PendingSubsetWrapper = () => {
            const [pendingSubset, setPendingSubset] = useState<DatasetSubset>('unassigned');

            const handleSubsetChange = (key: Key | null) => {
                if (key === 'validation' || key === 'testing' || key === 'training') {
                    setPendingSubset(key);
                }
            };

            return (
                <BottomToolbar
                    isReadOnlySubset={false}
                    mediaItem={mockMediaItem}
                    subset={pendingSubset}
                    onSubsetChange={handleSubsetChange}
                    hideHotkeys
                />
            );
        };

        render(
            <ZoomProvider>
                <AnnotationVisibilityProvider>
                    <CanvasSettingsProvider>
                        <PendingSubsetWrapper />
                    </CanvasSettingsProvider>
                </AnnotationVisibilityProvider>
            </ZoomProvider>
        );

        const pickerButton = screen.getByRole('button', { name: /select subset/i });
        fireEvent.click(pickerButton);

        const validationOption = screen.getByRole('option', { name: /Validation/i });
        fireEvent.click(validationOption);

        expect(screen.getByLabelText('Validation')).toBeInTheDocument();
    });
});
