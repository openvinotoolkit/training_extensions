// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType, SVGProps } from 'react';

import type { ToolType } from '../tool-type';

export interface ToolConfig {
    type: ToolType;
    icon: ComponentType<SVGProps<SVGSVGElement>>;
    hotkey: string;
    label: string;
    ariaLabel: string;
    tooltip?: {
        img?: string;
        description: string;
    };
}
