// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { useContainerSize } from './use-container-size';
import { ZoomTransform } from './zoom-transform';
import { ZoomProvider } from './zoom.provider';

rs.mock('./use-container-size', () => ({
    useContainerSize: rs.fn(),
}));

describe('Zoom', () => {
    it('Scales and translates content so that it fits the screen', () => {
        const screenSize = { width: 500, height: 500 };
        const contentSize = { width: 500, height: 500 };
        const expectedZoom = { translate: { x: 25, y: 25 }, scale: 0.9 };

        rs.mocked(useContainerSize).mockImplementation(() => screenSize);

        render(
            <ZoomProvider>
                <ZoomTransform target={contentSize}>Content</ZoomTransform>
            </ZoomProvider>
        );

        const transform = screen.getByTestId('zoom-transform');

        expect(transform).toHaveStyle({
            transform: `translate(${expectedZoom.translate.x}px, ${expectedZoom.translate.y}px) scale(${expectedZoom.scale})`,
        });
    });
});
