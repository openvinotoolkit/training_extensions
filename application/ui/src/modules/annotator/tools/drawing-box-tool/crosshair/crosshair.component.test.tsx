// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import type { Point } from '../../../../../shared/types';
import { Crosshair } from './crosshair.component';

describe('Crosshair', () => {
    const mockLocation: Point = { x: 100, y: 200 };

    it('renders correctly based on location', () => {
        const { container } = render(
            <svg>
                <Crosshair location={mockLocation} zoom={1} />
            </svg>
        );

        const group = container.querySelector('g');
        expect(group).toBeInTheDocument();

        const rects = container.querySelectorAll('rect');
        expect(rects).toHaveLength(2);

        const hasHorizontal = Array.from(rects).some((rect) => rect.getAttribute('width') === '100%');
        const hasVertical = Array.from(rects).some((rect) => rect.getAttribute('height') === '100%');
        expect(hasHorizontal).toBe(true);
        expect(hasVertical).toBe(true);

        const customLocation: Point = { x: 50, y: 75 };
        const { container: container2 } = render(
            <svg>
                <Crosshair location={customLocation} zoom={1} />
            </svg>
        );

        const [horizontal, vertical] = Array.from(container2.querySelectorAll('rect'));

        expect(horizontal).toHaveAttribute('y', '75');
        expect(horizontal).toHaveAttribute('width', '100%');
        expect(vertical).toHaveAttribute('x', '50');
        expect(vertical).toHaveAttribute('height', '100%');
    });

    it('updates when location changes', () => {
        const { container, rerender } = render(
            <svg>
                <Crosshair location={mockLocation} zoom={1} />
            </svg>
        );
        expect(container.querySelectorAll('rect')).toHaveLength(2);

        const newLocation: Point = { x: 150, y: 250 };

        rerender(
            <svg>
                <Crosshair location={newLocation} zoom={1} />
            </svg>
        );

        const [horizontal, vertical] = Array.from(container.querySelectorAll('rect'));

        expect(horizontal).toHaveAttribute('y', '250');
        expect(vertical).toHaveAttribute('x', '150');

        rerender(
            <svg>
                <Crosshair location={null} zoom={1} />
            </svg>
        );
        expect(container.querySelectorAll('rect')).toHaveLength(0);
    });
});
