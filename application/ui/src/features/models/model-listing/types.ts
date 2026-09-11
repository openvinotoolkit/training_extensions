// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';

export type GroupByMode = 'dataset' | 'architecture';

export type SortBy = 'name' | 'trained' | 'architecture' | 'dataset' | 'device' | 'size' | 'score';

export type SortDirection = 'asc' | 'desc';

export type SortDescriptor = {
    key: SortBy;
    direction: SortDirection;
};

export type DatasetGroup = {
    id: string;
    name: string;
    createdAt: string;
    labelCount: number;
    imageCount: number;
    trainingSubsets: {
        training: number;
        validation: number;
        testing: number;
    };
    filesDeleted: boolean;
};

export type ArchitectureGroup = {
    id: string;
};

export type GroupedModels = {
    group: DatasetGroup | ArchitectureGroup;
    models: Model[];
};
