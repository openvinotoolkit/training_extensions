// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType } from 'react';

import { useTool } from '../../tool-provider.component';
import type { ToolType } from '../../tool-type';
import { SegmentAnythingOptions } from '../segment-anything-tool/segment-anything-options.component';
import { SSIMOptions } from '../ssim-tool/ssim-options.component';

import classes from './tool-options.module.scss';

const TOOL_OPTIONS: Partial<Record<ToolType, ComponentType>> = {
    sam: SegmentAnythingOptions,
    ssim: SSIMOptions,
};

export const ToolOptions = () => {
    const { activeTool } = useTool();
    const Options = activeTool === null ? undefined : TOOL_OPTIONS[activeTool];

    if (Options === undefined) {
        return null;
    }

    return (
        <div role='group' aria-label='tool options' className={classes.toolOptions}>
            <Options />
        </div>
    );
};
