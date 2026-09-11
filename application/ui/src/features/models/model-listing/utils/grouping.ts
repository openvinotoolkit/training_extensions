// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision, Model } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { formatDateTime } from '../../../../shared/date-utils';
import type { GroupedModels } from '../types';

const formatDatasetStartTime = (dateString: string | null | undefined, t: TranslateFn): string => {
    const formatted = formatDateTime(dateString);

    return formatted === '-' ? formatted : t('dataset.revisions.createdOn', { date: formatted });
};

type GroupModelsByDatasetOptions = {
    datasetRevisions: DatasetRevision[];
};

export const groupModelsByDataset = (
    models: Model[],
    t: TranslateFn,
    options?: GroupModelsByDatasetOptions
): GroupedModels[] => {
    const { datasetRevisions = [] } = options || {};
    const groups: Record<string, GroupedModels> = {}; // datasetId -> models

    const datasetRevisionsMap = new Map(
        datasetRevisions.map((datasetRevision) => [datasetRevision.id, datasetRevision])
    );

    models.forEach((model) => {
        // NOTE: We only need this ?? check if we develop with seeded models. This id should always exist.
        const datasetId = model.training_info.dataset_revision_id ?? '';
        const labels = model.training_info.label_schema_revision?.labels;
        const labelCount = Array.isArray(labels) ? labels.length : 0;
        const datasetRevision = datasetRevisionsMap.get(datasetId);

        if (!groups[datasetId]) {
            groups[datasetId] = {
                group: {
                    id: datasetId,
                    name: datasetRevision?.name ?? t('dataset.revisions.unnamedName', { id: datasetId.slice(0, 8) }),
                    createdAt: formatDatasetStartTime(
                        datasetRevision ? datasetRevision.created_at : model.training_info.start_time,
                        t
                    ),
                    labelCount,
                    imageCount: datasetRevision?.item_counts?.total ?? 0,
                    trainingSubsets: {
                        training: datasetRevision?.item_counts?.training ?? 0,
                        validation: datasetRevision?.item_counts?.validation ?? 0,
                        testing: datasetRevision?.item_counts?.testing ?? 0,
                    },
                    filesDeleted: datasetRevision?.files_deleted ?? false,
                },
                models: [],
            };
        }

        if (groups[datasetId] !== undefined) {
            groups[datasetId].models.push(model);
        }
    });

    return Object.values(groups);
};

export const groupModelsByArchitecture = (models: Model[]): GroupedModels[] => {
    const groups: Record<string, GroupedModels> = {}; // architecture -> models

    models.forEach((model) => {
        const arch = model.architecture ?? 'Unknown';

        if (!groups[arch]) {
            groups[arch] = {
                group: {
                    id: arch,
                },
                models: [],
            };
        }

        groups[arch].models.push(model);
    });

    return Object.values(groups);
};
