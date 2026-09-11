// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';

export type TaskOption = {
    id: string;
    imageSrc: string;
    title: string;
    ariaLabel: string;
    description: string;
    advice: string;
    verb: string;
    value: TaskType;
};
