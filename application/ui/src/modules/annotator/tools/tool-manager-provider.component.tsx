// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PropsWithChildren, ReactNode } from 'react';

import { useAvailableTools } from './annotator-tools/use-available-tools';
import { PolygonStateProvider } from './polygon-tool/polygon-state-provider.component';
import { SSIMStateProvider } from './ssim-tool/ssim-state-provider.component';

/**
 * Composes per-tool state providers that need to sit above both the primary
 * toolbar and the annotator canvas (so the toolbar's undo/redo buttons can
 * reach tool-local undo stacks while a tool is active).
 *
 * Adding state for a new tool: nest an additional provider here.
 * The composition shape is stable per project (task type does not change
 * mid-session), so the React subtree remains stable and the toolbar and
 * canvas are never unnecessarily remounted.
 */
export const ToolManagerProvider = ({ children }: PropsWithChildren) => {
    const availableTools = useAvailableTools();
    const hasPolygonDrawingTools = availableTools.some(
        (tool) => tool.type === 'polygon' || tool.type === 'magnetic-lasso'
    );
    const hasSSIMTool = availableTools.some((tool) => tool.type === 'ssim');

    let content: ReactNode = children;

    if (hasSSIMTool) {
        content = <SSIMStateProvider>{content}</SSIMStateProvider>;
    }

    if (hasPolygonDrawingTools) {
        content = <PolygonStateProvider>{content}</PolygonStateProvider>;
    }

    return <>{content}</>;
};
