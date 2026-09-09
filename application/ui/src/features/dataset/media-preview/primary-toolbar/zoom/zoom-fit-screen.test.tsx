// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ZoomFitScreen } from './zoom-fit-screen.component';

const mockedFitToScreen = vi.fn();

vi.mock(import('@/components/zoom/zoom.provider'), async (importOriginal) => {
    const actual = await importOriginal();
    return {
        ...actual,
        useSetZoom: () => ({
            setZoom: vi.fn(),
            fitToScreen: mockedFitToScreen,
            onZoomChange: vi.fn(),
        }),
    };
});

describe('ZoomFitScreen', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('calls fitToScreen when button is clicked', () => {
        render(<ZoomFitScreen />);

        fireEvent.click(screen.getByRole('button', { name: /fit to screen/i }));

        expect(mockedFitToScreen).toHaveBeenCalledTimes(1);
    });
});
