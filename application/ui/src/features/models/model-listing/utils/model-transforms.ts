// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision, Model } from '@/api/types';

import type { GroupByMode, GroupedModels, SortDescriptor } from '../types';
import { groupModelsByArchitecture, groupModelsByDataset } from './grouping';
import { DEFAULT_SORT, sortModels } from './sorting';
import { isFailedModel, isTrainingModel } from './utils';

export const filterBySearch = (models: Model[], query: string): Model[] =>
    query ? models.filter((model) => model.name.toLowerCase().includes(query.toLowerCase())) : models;

export const filterOutFailedModels = (models: Model[]): Model[] => {
    return models.filter((model) => !isFailedModel(model));
};

export const filterOutTrainingModels = (models: Model[]): Model[] => {
    return models.filter((model) => !isTrainingModel(model));
};

export const groupModels = (
    models: Model[],
    mode: GroupByMode,
    datasetRevisions: DatasetRevision[]
): GroupedModels[] =>
    mode === 'dataset' ? groupModelsByDataset(models, { datasetRevisions }) : groupModelsByArchitecture(models);

export const sortGroupedModels = (
    groups: GroupedModels[],
    sortByGroup: Record<string, SortDescriptor>,
    datasetRevisions: DatasetRevision[]
): GroupedModels[] =>
    groups.map((group) => {
        const { key, direction } = sortByGroup[group.group.id] ?? DEFAULT_SORT;

        return {
            ...group,
            models: sortModels(group.models, key, datasetRevisions, direction),
        };
    });

export const removeEmpty = (groups: GroupedModels[]): GroupedModels[] =>
    groups.filter((group) => group.models.length > 0);
