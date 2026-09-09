// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '@/components/zoom/zoom.provider';
import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ZoomSelector } from './zoom-selector.component';

const mockedOnZoomChange = vi.fn();

vi.mock(import('@/components/zoom/zoom.provider'), async (importOriginal) => {
    const actual = await importOriginal();
    return {
        ...actual,
        useSetZoom: () => ({
            setZoom: vi.fn(),
            fitToScreen: vi.fn(),
            onZoomChange: mockedOnZoomChange,
        }),
        useZoom: vi.fn(),
    };
});

describe('ZoomSelector', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('displays the correct zoom value', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 0.21,
            maxZoomIn: 0,
            hasAnimation: false,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);
        expect(screen.getByText('21%')).toBeVisible();
    });

    it('calls onZoomChange when zoom in button is clicked', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 0.5,
            maxZoomIn: 2,
            hasAnimation: false,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);
        const zoomInButton = screen.getByRole('button', { name: /zoom in/i });
        fireEvent.click(zoomInButton);
        expect(mockedOnZoomChange).toHaveBeenCalledWith(1);
    });

    it('calls onZoomChange when zoom out button is clicked', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 1,
            maxZoomIn: 2,
            hasAnimation: false,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);
        const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });
        fireEvent.click(zoomOutButton);
        expect(mockedOnZoomChange).toHaveBeenCalledWith(-1);
    });

    it('disables zoom in button at max zoom in', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 2,
            maxZoomIn: 2,
            hasAnimation: false,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0, x: 0, y: 0 },
        });

        render(<ZoomSelector />);

        expect(screen.getByRole('button', { name: /zoom in/i })).toBeDisabled();
    });

    it('disables zoom out button at max zoom out', () => {
        vi.mocked(useZoom).mockReturnValue({
            scale: 0.1,
            maxZoomIn: 2,
            hasAnimation: false,
            translate: { x: 0, y: 0 },
            initialCoordinates: { scale: 0.1, x: 0, y: 0 },
        });

        render(<ZoomSelector />);

        expect(screen.getByRole('button', { name: /zoom out/i })).toBeDisabled();
    });
});
