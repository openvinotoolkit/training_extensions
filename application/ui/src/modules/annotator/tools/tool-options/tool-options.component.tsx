// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTool } from '../../tool-provider.component';
import { SSIMOptions } from '../ssim-tool/ssim-options.component';

import classes from './tool-options.module.scss';

export const ToolOptions = () => {
    const { activeTool } = useTool();

    if (activeTool !== 'ssim') {
        return null;
    }

    return (
        <div role='group' aria-label='tool options' className={classes.toolOptions}>
            <SSIMOptions />
        </div>
    );
};
