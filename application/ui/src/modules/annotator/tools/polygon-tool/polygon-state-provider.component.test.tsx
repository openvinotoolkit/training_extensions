// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { UndoRedoProvider, useUndoRedo } from '../../../../shared/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../../../shared/undo-redo/use-undo-redo-state';
import type { ToolType } from '../../tool-type';
import { PolygonStateProvider, usePolygonState } from './polygon-state-provider.component';

const mockSetActiveTool = rs.fn();
let mockActiveTool: ToolType | null = 'polygon';

rs.mock('../../tool-provider.component', () => ({
    useTool: () => ({ activeTool: mockActiveTool, setActiveTool: mockSetActiveTool }),
}));

const PolygonStateConsumer = () => {
    const { segments, pointerLine, lassoSegment, setSegments, setPointerLine, setLassoSegment, undoRedoActions } =
        usePolygonState();

    return (
        <div>
            <span data-testid='segments'>{JSON.stringify(segments)}</span>
            <span data-testid='pointer-line'>{JSON.stringify(pointerLine)}</span>
            <span data-testid='lasso-segment'>{JSON.stringify(lassoSegment)}</span>

            <button onClick={() => setSegments((prev) => [...prev, [{ x: 1, y: 1 }]])}>Add segment</button>
            <button onClick={() => setPointerLine([{ x: 2, y: 2 }])}>Set pointer line</button>
            <button onClick={() => setLassoSegment([{ x: 3, y: 3 }])}>Set lasso segment</button>

            <button onClick={undoRedoActions.undo} disabled={!undoRedoActions.canUndo}>
                Undo
            </button>
            <button onClick={undoRedoActions.redo} disabled={!undoRedoActions.canRedo}>
                Redo
            </button>
        </div>
    );
};

const ToolbarUndoRedoConsumer = () => {
    const { undo, redo, canUndo, canRedo } = useUndoRedo();

    return (
        <div>
            <button onClick={undo} disabled={!canUndo}>
                Toolbar undo
            </button>
            <button onClick={redo} disabled={!canRedo}>
                Toolbar redo
            </button>
        </div>
    );
};

const Session = ({ children }: { children: ReactNode }) => {
    const [, , history] = useUndoRedoState(0);

    return <UndoRedoProvider baseHistory={history}>{children}</UndoRedoProvider>;
};

describe('PolygonStateProvider', () => {
    beforeEach(() => {
        mockActiveTool = 'polygon';
    });

    const renderProvider = () =>
        render(
            <Session>
                <PolygonStateProvider>
                    <PolygonStateConsumer />
                    <ToolbarUndoRedoConsumer />
                </PolygonStateProvider>
            </Session>
        );

    it('toolbar undo removes the last segment', () => {
        renderProvider();

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(2);

        fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));
        expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(1);
    });

    it('toolbar undo clears pointerLine and lassoSegment (fixes frozen-tool bug)', () => {
        renderProvider();

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set pointer line' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set lasso segment' }));

        expect(screen.getByTestId('pointer-line')).not.toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).not.toHaveTextContent('[]');

        fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));

        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });

    it('toolbar redo clears pointerLine and lassoSegment', () => {
        renderProvider();

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));

        // Set transient state after undo to simulate pointer activity
        fireEvent.click(screen.getByRole('button', { name: 'Set pointer line' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set lasso segment' }));

        fireEvent.click(screen.getByRole('button', { name: 'Toolbar redo' }));

        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });

    it('resets all drawing state when activeTool changes', () => {
        const tree = (
            <Session>
                <PolygonStateProvider>
                    <PolygonStateConsumer />
                    <ToolbarUndoRedoConsumer />
                </PolygonStateProvider>
            </Session>
        );
        const { rerender } = render(tree);

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set pointer line' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set lasso segment' }));

        expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(1);
        expect(screen.getByTestId('pointer-line')).not.toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).not.toHaveTextContent('[]');

        // Simulate user switching to a different tool
        mockActiveTool = 'magnetic-lasso';
        rerender(tree);

        expect(screen.getByTestId('segments')).toHaveTextContent('[]');
        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });
});
