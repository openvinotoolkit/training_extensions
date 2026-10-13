// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';
import { render, screen } from '@testing-library/react';

import type { ToolType } from '../../tool-type';
import { SSIMStateProvider, useSSIMState } from './ssim-state-provider.component';
import { useSSIM } from './use-ssim.hook';

let mockActiveTool: ToolType | null = 'ssim';
let mockMediaItem = { id: 'media-1' } as Media;
let mockTaskType = 'detection';
const mockReset = vi.fn();

vi.mock('../../tool-provider.component', () => ({
    useTool: () => ({ activeTool: mockActiveTool, setActiveTool: vi.fn() }),
}));

vi.mock('../../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({ mediaItem: mockMediaItem }),
}));

vi.mock('hooks/use-project-task.hook', () => ({
    useProjectTask: () => mockTaskType,
}));

vi.mock('./use-ssim.hook', () => ({
    useSSIM: vi.fn(() => ({ reset: mockReset })),
}));

const Consumer = () => {
    useSSIMState();

    return <span>consumer</span>;
};

const renderTree = () => (
    <SSIMStateProvider>
        <Consumer />
    </SSIMStateProvider>
);

describe('SSIMStateProvider', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockActiveTool = 'ssim';
        mockMediaItem = { id: 'media-1' } as Media;
        mockTaskType = 'detection';
    });

    it.each([
        ['detection', 'rectangle'],
        ['instance_segmentation', 'polygon'],
    ])('uses the %s task to derive the %s shape type', (taskType, shapeType) => {
        mockTaskType = taskType;

        render(renderTree());

        expect(screen.getByText('consumer')).toBeInTheDocument();
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
});
