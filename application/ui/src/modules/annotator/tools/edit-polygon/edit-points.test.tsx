// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import '@wessberg/pointer-events';

import { ReactNode } from 'react';

import { ThemeProvider } from '@geti-ui/ui';
import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { Polygon } from '../../../../shared/types';
import { EditPoints } from './edit-points.component';

const renderEditPoints = (element: ReactNode) => {
    render(
        <ThemeProvider>
            <svg>{element}</svg>
        </ThemeProvider>
    );
};

describe('EditPoints', () => {
    const polygon: Polygon = {
        type: 'polygon',
        points: [
            { x: 20, y: 10 },
            { x: 70, y: 30 },
            { x: 80, y: 90 },
        ],
    };

    it('deletes a single point', async () => {
        const mockRemovePoints = rs.fn();

        renderEditPoints(
            <EditPoints
                zoom={1}
                shape={polygon}
                addPoint={rs.fn()}
                onComplete={rs.fn()}
                moveAnchorTo={rs.fn()}
                removePoints={mockRemovePoints}
            />
        );

        const pointIndex = 0;
        const point = screen.getByLabelText(`Click to select point ${pointIndex}`);
        expect(point).toHaveAttribute('aria-selected', 'false');

        fireEvent.pointerDown(point);

        expect(point).toHaveAttribute('aria-selected', 'true');

        fireEvent.keyDown(document.body, { key: 'Delete', keyCode: 46, code: 'Delete' });
        expect(mockRemovePoints).toHaveBeenLastCalledWith([pointIndex]);
    });

    it('selects and deletes multiple points', async () => {
        const mockRemovePoints = rs.fn();

        renderEditPoints(
            <EditPoints
                zoom={1}
                shape={polygon}
                addPoint={rs.fn()}
                onComplete={rs.fn()}
                moveAnchorTo={rs.fn()}
                removePoints={mockRemovePoints}
            />
        );
        const pointOneIndex = 0;
        const pointTwoIndex = 1;
        const pointOne = screen.getByLabelText(`Click to select point ${pointOneIndex}`);
        const pointTwo = screen.getByLabelText(`Click to select point ${pointTwoIndex}`);
        expect(pointOne).toHaveAttribute('aria-selected', 'false');
        expect(pointTwo).toHaveAttribute('aria-selected', 'false');

        fireEvent.pointerDown(pointOne, { shiftKey: true });
        fireEvent.pointerDown(pointTwo, { shiftKey: true });

        expect(pointOne).toHaveAttribute('aria-selected', 'true');
        expect(pointTwo).toHaveAttribute('aria-selected', 'true');

        fireEvent.keyDown(document.body, { key: 'Delete', keyCode: 46, code: 'Delete' });
        expect(mockRemovePoints).toHaveBeenLastCalledWith([pointOneIndex, pointTwoIndex]);
    });
});
