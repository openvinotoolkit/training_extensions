// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision, Model } from '@/api/types';
import dayjs from 'dayjs';
import { orderBy } from 'lodash-es';

import { getTestingMetric } from '../components/model-row/utils';
import type { GroupedModels, SortBy, SortDescriptor, SortDirection } from '../types';

export const DEFAULT_SORT_DIRECTIONS: Record<SortBy, SortDirection> = {
    name: 'asc',
    trained: 'desc',
    architecture: 'asc',
    dataset: 'desc',
    device: 'asc',
    size: 'asc',
    score: 'desc',
};

export const DEFAULT_SORT: SortDescriptor = { key: 'score', direction: 'desc' };

export const sortModels = (
    models: Model[],
    sortBy: SortBy,
    datasetRevisions: DatasetRevision[],
    direction: SortDirection = DEFAULT_SORT_DIRECTIONS[sortBy]
): Model[] => {
    switch (sortBy) {
        case 'name':
            return orderBy(models, (model) => model.name.toLowerCase(), direction);
        case 'architecture':
            return orderBy(models, (model) => model.architecture?.toLowerCase() ?? '', direction);
        case 'trained': {
            const endTimes = new Map(models.map((model) => [model, dayjs(model.training_info?.end_time)]));

            return orderBy(
                models,
                [
                    // Models without a valid training date come last in both directions.
                    (model) => (endTimes.get(model)?.isValid() ? 0 : 1),
                    (model) => {
                        const date = endTimes.get(model);

                        return date?.isValid() ? date.valueOf() : 0;
                    },
                ],
                ['asc', direction]
            );
        }
        case 'device':
            return orderBy(
                models,
                [
                    // Models without a device come last.
                    (model) => (model.training_info?.device?.name != null ? 0 : 1),
                    (model) => model.training_info?.device?.name?.toLowerCase() ?? '',
                ],
                ['asc', direction]
            );
        case 'size':
            return orderBy(models, (model) => model.size ?? 0, direction);
        case 'score': {
            const metrics = new Map(models.map((model) => [model, getTestingMetric(model)]));

            return orderBy(
                models,
                [
                    // Models without a score come last.
                    (model) => (metrics.get(model) !== undefined ? 0 : 1),
                    (model) => metrics.get(model)?.value ?? 0,
                ],
                ['asc', direction]
            );
        }
        case 'dataset': {
            const datasetRevisionsMap = new Map(
                datasetRevisions.map((datasetRevision) => [datasetRevision.id, datasetRevision])
            );

            const revisions = new Map(
                models.map((model) => {
                    const id = model.training_info?.dataset_revision_id;

                    return [model, id != null ? datasetRevisionsMap.get(id) : undefined];
                })
            );

            return orderBy(
                models,
                [
                    // First: models without a resolvable dataset revision come last.
                    (model) => (revisions.get(model) != null ? 0 : 1),
                    // Second: sort by dataset revision creation date.
                    (model) => revisions.get(model)?.created_at ?? '',
                    // Third: sort by dataset revision name.
                    (model) => revisions.get(model)?.name?.toLowerCase() ?? '',
                ],
                ['asc', direction, direction]
            );
        }
        default:
            console.error(`Unknown sort option: ${sortBy satisfies never}`);
            return models;
    }
};

export const sortGroupedModelsByDatasetRevisionDate = (
    groupedModels: GroupedModels[],
    datasetRevisions: DatasetRevision[]
): GroupedModels[] => {
    const datasetRevisionsMap = new Map(
        datasetRevisions.map((datasetRevision) => [datasetRevision.id, datasetRevision])
    );

    return orderBy(
        groupedModels,
        (group) => {
            const mostRecentModel = sortModels(group.models, 'dataset', datasetRevisions)?.at(0);
            const modelDatasetRevisionId = mostRecentModel?.training_info?.dataset_revision_id;
            const datasetRevisionDate =
                modelDatasetRevisionId != null ? datasetRevisionsMap.get(modelDatasetRevisionId)?.created_at : null;

            return datasetRevisionDate ?? '';
        },
        'desc'
    );
};
