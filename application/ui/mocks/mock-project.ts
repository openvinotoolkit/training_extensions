// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Project } from '@/api/types';

export const getMockedProject = (customProject: Partial<Project> = {}): Project => {
    return {
        id: '7b073838-99d3-42ff-9018-4e901eb047fc',
        name: 'animals',
        task: {
            exclusive_labels: true,
            labels: [
                {
                    color: '#FF5733',
                    hotkey: 'S',
                    id: 'a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29',
                    name: 'Object',
                },
            ],
            task_type: 'detection',
        },
        active_pipeline: false,
        created_at: '2024-10-01T12:00:00Z',
        ...customProject,
    };
};
