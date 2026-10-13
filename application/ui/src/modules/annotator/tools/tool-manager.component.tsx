// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTool } from '../tool-provider.component';
import { BoundingBoxTool } from './bounding-box-tool/bounding-box-tool.component';
import { MagneticLasso } from './magnetic-lasso/magnetic-lasso.component';
import { PolygonTool } from './polygon-tool/polygon-tool.component';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';
import { SSIMTool } from './ssim-tool/ssim-tool.component';
import { usePreloadWebworkers } from './use-preload-webworkers.hook';

export const ToolManager = () => {
    const { activeTool } = useTool();

    // Preload workers when the tool manager is mounted, so that "smart" tool is ready
    // to use as soon as the user selects them. Not a huge performance gain but
    // it helps a bit.
    usePreloadWebworkers();

    if (activeTool === 'bounding-box') {
        return <BoundingBoxTool />;
    }

    if (activeTool === 'sam') {
        return <SegmentAnythingTool />;
    }

    if (activeTool === 'polygon') {
        return <PolygonTool />;
    }

    if (activeTool === 'magnetic-lasso') {
        return <MagneticLasso />;
    }

    if (activeTool === 'ssim') {
        return <SSIMTool />;
    }

    return null;
};
