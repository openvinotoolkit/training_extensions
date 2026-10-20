// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import * as actualZoom from '@/components/zoom/zoom.provider' with { rstest: 'importActual' };
import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ZoomFitScreen } from './zoom-fit-screen.component';

const mockedFitToScreen = rs.fn();

rs.mock('@/components/zoom/zoom.provider', () => ({
    ...actualZoom,
    useSetZoom: () => ({
        setZoom: rs.fn(),
        fitToScreen: mockedFitToScreen,
        onZoomChange: rs.fn(),
    }),
}));

describe('ZoomFitScreen', () => {
    beforeEach(() => {
        rs.clearAllMocks();
    });

    it('calls fitToScreen when button is clicked', () => {
        render(<ZoomFitScreen />);

        fireEvent.click(screen.getByRole('button', { name: /fit to screen/i }));

        expect(mockedFitToScreen).toHaveBeenCalledTimes(1);
    });
});
