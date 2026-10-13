// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PropsWithChildren, ReactNode, useEffect } from 'react';

import { useProjectTask } from 'hooks/use-project-task.hook';

import { getDefaultTool, useTool } from '../tool-provider.component';
import { useAvailableTools, useTaskTools } from './annotator-tools/use-available-tools';
import { PolygonStateProvider } from './polygon-tool/polygon-state-provider.component';
import { SSIMStateProvider } from './ssim-tool/ssim-state-provider.component';

// Smart tools are unavailable for oversized media, so the active tool may have to change with the media item.
const useFallbackToAvailableTool = () => {
    const taskType = useProjectTask();
    const availableTools = useAvailableTools();
    const { activeTool, setActiveTool } = useTool();
    const isActiveToolAvailable = activeTool === null || availableTools.some((tool) => tool.type === activeTool);

    useEffect(() => {
        if (!isActiveToolAvailable) {
            setActiveTool(getDefaultTool(taskType));
        }
    }, [isActiveToolAvailable, setActiveTool, taskType]);
};

/**
 * Composes per-tool state providers that need to sit above both the primary
 * toolbar and the annotator canvas (so the toolbar's undo/redo buttons can
 * reach tool-local undo stacks while a tool is active).
 *
 * Adding state for a new tool: nest an additional provider here.
 * The composition is derived from the task's tools (not the media-filtered
 * ones), so the React subtree remains stable and the toolbar and canvas are
 * never unnecessarily remounted.
 */
export const ToolManagerProvider = ({ children }: PropsWithChildren) => {
    const taskTools = useTaskTools();
    const hasPolygonDrawingTools = taskTools.some((tool) => tool.type === 'polygon' || tool.type === 'magnetic-lasso');
    const hasSSIMTool = taskTools.some((tool) => tool.type === 'ssim');

    useFallbackToAvailableTool();

    let content: ReactNode = children;

    if (hasSSIMTool) {
        content = <SSIMStateProvider>{content}</SSIMStateProvider>;
    }

    if (hasPolygonDrawingTools) {
        content = <PolygonStateProvider>{content}</PolygonStateProvider>;
    }

    return <>{content}</>;
};
