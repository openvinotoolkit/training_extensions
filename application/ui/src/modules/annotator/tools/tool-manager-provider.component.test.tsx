// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PropsWithChildren } from 'react';

import { render, screen } from '@testing-library/react';

import * as actualToolProvider from '../tool-provider.component' with { rstest: 'importActual' };
import type { ToolType } from '../tool-type';
import type { ToolConfig } from './interface';
import { ToolManagerProvider } from './tool-manager-provider.component';

const toolConfig = (type: ToolType) => ({ type }) as ToolConfig;
const DETECTION_TOOLS = [toolConfig('selection'), toolConfig('bounding-box'), toolConfig('sam'), toolConfig('ssim')];

let mockAvailableTools: ToolConfig[] = DETECTION_TOOLS;
let mockActiveTool: ToolType | null = 'ssim';
const mockSetActiveTool = rs.fn();

rs.mock('hooks/use-project-task.hook', () => ({
    useProjectTask: () => 'detection',
}));

rs.mock('../tool-provider.component', () => ({
    ...actualToolProvider,
    useTool: () => ({ activeTool: mockActiveTool, setActiveTool: mockSetActiveTool }),
}));

rs.mock('./annotator-tools/use-available-tools', () => ({
    useTaskTools: () => DETECTION_TOOLS,
    useAvailableTools: () => mockAvailableTools,
}));

rs.mock('./ssim-tool/ssim-state-provider.component', () => ({
    SSIMStateProvider: ({ children }: PropsWithChildren) => <div data-testid='ssim-state'>{children}</div>,
}));

const renderProvider = () => (
    <ToolManagerProvider>
        <span>content</span>
    </ToolManagerProvider>
);

describe('ToolManagerProvider', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        mockAvailableTools = DETECTION_TOOLS;
        mockActiveTool = 'ssim';
    });

    it('keeps the SSIM state mounted and falls back to the default tool when SSIM becomes unavailable', () => {
        const { rerender } = render(renderProvider());

        expect(screen.getByTestId('ssim-state')).toHaveTextContent('content');
        expect(mockSetActiveTool).not.toHaveBeenCalled();

        mockAvailableTools = DETECTION_TOOLS.filter((tool) => tool.type !== 'sam' && tool.type !== 'ssim');
        rerender(renderProvider());

        expect(screen.getByTestId('ssim-state')).toHaveTextContent('content');
        expect(mockSetActiveTool).toHaveBeenCalledWith('bounding-box');
    });

    it('keeps the active tool when it is still available', () => {
        mockActiveTool = 'bounding-box';
        const { rerender } = render(renderProvider());

        mockAvailableTools = DETECTION_TOOLS.filter((tool) => tool.type !== 'sam' && tool.type !== 'ssim');
        rerender(renderProvider());

        expect(mockSetActiveTool).not.toHaveBeenCalled();
    });
});
