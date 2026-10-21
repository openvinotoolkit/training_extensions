// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label, Media } from '@/api/types';
import { fireEvent, render, screen } from '@testing-library/react';

import type { Shape } from '../../../../shared/types';
import type { AnnotatorMode } from '../../annotator-mode';
import type { ToolType } from '../../tool-type';
import { SSIMStateProvider, useSSIMState } from './ssim-state-provider.component';
import { useSSIM } from './use-ssim.hook';

const template: Shape = { type: 'rectangle', x: 0, y: 0, width: 10, height: 10 };
const match: Shape = { type: 'rectangle', x: 50, y: 50, width: 10, height: 10 };
const label = { id: 'label-1', name: 'Red' } as Label;

let mockActiveTool: ToolType | null = 'ssim';
let mockMediaItem = { id: 'media-1' } as Media;
let mockTaskType = 'detection';
let mockMode: AnnotatorMode = 'annotation';
let mockShapes: Shape[] = [];
const mockReset = rs.fn();
const mockUpdateToolState = rs.fn();
const mockRerun = rs.fn();
const mockAddAndSelectAnnotations = rs.fn();

rs.mock('../../tool-provider.component', () => ({
    useTool: () => ({ activeTool: mockActiveTool, setActiveTool: rs.fn() }),
}));

rs.mock('../../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({ mediaItem: mockMediaItem }),
}));

rs.mock('../../annotation-document-provider.component', () => ({
    useAnnotations: () => ({ mode: mockMode }),
}));

rs.mock('../../annotator-labels-provider.component', () => ({
    useAnnotatorLabels: () => ({ selectedLabel: label }),
}));

rs.mock('../use-add-and-select-annotations.hook', () => ({
    useAddAndSelectAnnotations: () => ({ addAndSelectAnnotations: mockAddAndSelectAnnotations }),
}));

rs.mock('hooks/use-project-task.hook', () => ({
    useProjectTask: () => mockTaskType,
}));

rs.mock('./use-ssim.hook', () => ({
    useSSIM: rs.fn(() => ({
        reset: mockReset,
        updateToolState: mockUpdateToolState,
        rerun: mockRerun,
        toolState: { shapes: mockShapes },
    })),
}));

const Consumer = () => {
    const { hasPreview, accept, reject, setNumberOfItems, autoMergeDuplicates, setAutoMergeDuplicates } =
        useSSIMState();

    return (
        <>
            <span>{hasPreview ? 'has preview' : 'no preview'}</span>
            <span>{autoMergeDuplicates ? 'merging duplicates' : 'keeping duplicates'}</span>
            <button onClick={accept}>Accept</button>
            <button onClick={reject}>Reject</button>
            <button onClick={() => setNumberOfItems(3)}>Show 3 items</button>
            <button onClick={() => setAutoMergeDuplicates(false)}>Keep duplicates</button>
        </>
    );
};

const renderTree = () => (
    <SSIMStateProvider>
        <Consumer />
    </SSIMStateProvider>
);

describe('SSIMStateProvider', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        mockActiveTool = 'ssim';
        mockMediaItem = { id: 'media-1' } as Media;
        mockTaskType = 'detection';
        mockMode = 'annotation';
        mockShapes = [];
    });

    it.each([
        ['detection', 'rectangle'],
        ['instance_segmentation', 'polygon'],
    ])('uses the %s task to derive the %s shape type', (taskType, shapeType) => {
        mockTaskType = taskType;

        render(renderTree());

        expect(screen.getByText('no preview')).toBeInTheDocument();
        expect(useSSIM).toHaveBeenCalledWith(shapeType);
    });

    it('resets the state when the active tool changes', () => {
        const { rerender } = render(renderTree());
        mockReset.mockClear();

        mockActiveTool = 'bounding-box';
        rerender(renderTree());

        expect(mockReset).toHaveBeenCalledTimes(1);
    });

    it('resets the state when the media item changes', () => {
        const { rerender } = render(renderTree());
        mockReset.mockClear();

        mockMediaItem = { id: 'media-2' } as Media;
        rerender(renderTree());

        expect(mockReset).toHaveBeenCalledTimes(1);
    });

    it('does not reset the state on unrelated re-renders', () => {
        const { rerender } = render(renderTree());
        mockReset.mockClear();

        rerender(renderTree());

        expect(mockReset).not.toHaveBeenCalled();
    });

    it('resets the state when the annotator mode changes', () => {
        const { rerender } = render(renderTree());
        mockReset.mockClear();

        mockMode = 'prediction';
        rerender(renderTree());

        expect(mockReset).toHaveBeenCalledTimes(1);
    });

    it('adds the previewed shapes with the selected label and resets on accept', () => {
        mockShapes = [template, match];
        render(renderTree());
        mockReset.mockClear();

        expect(screen.getByText('has preview')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Accept' }));

        expect(mockAddAndSelectAnnotations).toHaveBeenCalledWith([template, match], [label]);
        expect(mockReset).toHaveBeenCalledTimes(1);
    });

    it('discards the preview without adding annotations on reject', () => {
        mockShapes = [template, match];
        render(renderTree());
        mockReset.mockClear();

        fireEvent.click(screen.getByRole('button', { name: 'Reject' }));

        expect(mockAddAndSelectAnnotations).not.toHaveBeenCalled();
        expect(mockReset).toHaveBeenCalledTimes(1);
    });

    it('updates the number of previewed items', () => {
        render(renderTree());

        fireEvent.click(screen.getByRole('button', { name: 'Show 3 items' }));

        expect(mockUpdateToolState).toHaveBeenCalledWith({ threshold: 3 });
    });

    it('merges duplicates by default and reruns the last template when toggled', () => {
        render(renderTree());

        expect(screen.getByText('merging duplicates')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Keep duplicates' }));

        expect(screen.getByText('keeping duplicates')).toBeInTheDocument();
        expect(mockRerun).toHaveBeenCalledWith({ autoMergeDuplicates: false });
    });
});
