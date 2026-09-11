// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType, SVGProps } from 'react';

export type ToolType = 'selection' | 'bounding-box' | 'polygon' | 'sam' | 'magnetic-lasso' | 'ssim';

export interface ToolConfig {
    type: ToolType;
    icon: ComponentType<SVGProps<SVGSVGElement>>;
    hotkey: string;
    label: string;
    ariaLabel: string;
    tooltip?: {
        img: string;
        description: string;
    };
}
