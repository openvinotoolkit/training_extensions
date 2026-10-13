// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import type { Point } from '../../../../../shared/types';
import { CrosshairLine } from './crosshair-line.component';

describe('CrosshairLine', () => {
    const mockPoint: Point = { x: 100, y: 200 };

    it('renders rect with correct direction attributes', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={1} point={mockPoint} direction='horizontal' />
            </svg>
        );
        const hRect = container.querySelector('rect');
        expect(hRect).toBeInTheDocument();
        expect(hRect).toHaveAttribute('y', '200');
        expect(hRect).toHaveAttribute('width', '100%');
        expect(hRect).toHaveAttribute('height', '1');

        const { container: vContainer } = render(
            <svg>
                <CrosshairLine zoom={1} point={mockPoint} direction='vertical' />
            </svg>
        );
        const vRect = vContainer.querySelector('rect');
        expect(vRect).toHaveAttribute('x', '100');
        expect(vRect).toHaveAttribute('width', '1');
        expect(vRect).toHaveAttribute('height', '100%');
    });

    it.each([
        { zoom: 2, direction: 'horizontal', attribute: 'height', expected: '0.5' },
        { zoom: 4, direction: 'vertical', attribute: 'width', expected: '0.25' },
        { zoom: 0.5, direction: 'horizontal', attribute: 'height', expected: '2' },
    ] as const)(
        'scales $direction line thickness inversely to zoom $zoom',
        ({ zoom, direction, attribute, expected }) => {
            const { container } = render(
                <svg>
                    <CrosshairLine zoom={zoom} point={mockPoint} direction={direction} />
                </svg>
            );
            const rect = container.querySelector('rect');
            expect(rect).toHaveAttribute(attribute, expected);
            expect(rect).toHaveAttribute('stroke-width', expected);
        }
    );

    it('applies correct styling', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={1} point={mockPoint} direction='horizontal' />
            </svg>
        );
        const rect = container.querySelector('rect');
        expect(rect).toHaveAttribute('fill', 'white');
        expect(rect).toHaveAttribute('fill-opacity', '0.9');
        expect(rect).toHaveAttribute('stroke', '#000000');
        expect(rect).toHaveAttribute('stroke-opacity', '0.12');
        expect(rect).toHaveAttribute('stroke-width', '1');
    });
});
