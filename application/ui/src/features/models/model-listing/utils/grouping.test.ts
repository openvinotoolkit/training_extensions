// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision } from '@/api/types';
import { createI18nInstance } from '@/i18n';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';

import type { DatasetGroup } from '../types';
import { groupModelsByArchitecture, groupModelsByDataset } from './grouping';

const { t } = createI18nInstance({ lng: 'en' });

describe('groupModelsByDataset', () => {
    it('should return empty array for empty models', () => {
        const result = groupModelsByDataset([], t);

        expect(result).toEqual([]);
    });

    it('should group models by dataset_revision_id', () => {
        const models = [
            getMockedModel({
                id: 'model-1',
                training_info: {
                    status: 'successful',
                    dataset_revision_id: 'dataset-1',
                    label_schema_revision: {},
                },
            }),
            getMockedModel({
                id: 'model-2',
                training_info: {
                    status: 'successful',
                    dataset_revision_id: 'dataset-1',
                    label_schema_revision: {},
                },
            }),
            getMockedModel({
                id: 'model-3',
                training_info: {
                    status: 'successful',
                    dataset_revision_id: 'dataset-2',
                    label_schema_revision: {},
                },
            }),
        ];

        const groupedModels = groupModelsByDataset(models, t, {
            datasetRevisions: [
                getMockedDatasetRevision({ id: 'dataset-1' }),
                getMockedDatasetRevision({ id: 'dataset-2' }),
            ],
        });

        expect(groupedModels).toHaveLength(2);
        expect(groupedModels[0].models).toHaveLength(2);
        expect(groupedModels[1].models).toHaveLength(1);
    });

    it('should use dataset revision names when provided', () => {
        const models = [
            getMockedModel({
                id: 'model-1',
                training_info: {
                    status: 'successful',
                    dataset_revision_id: 'dataset-1',
                    label_schema_revision: {},
                },
            }),
        ];

        const datasetRevisions = [
            getMockedDatasetRevision({
                id: 'dataset-1',
                created_at: '2025-01-01T00:00:00.000000+00:00',
                name: 'My Custom Dataset',
                files_deleted: false,
                item_counts: {
                    total: 100,
                    training: 70,
                    validation: 20,
                    testing: 10,
                },
            }),
        ];

        const groupedModels = groupModelsByDataset(models, t, { datasetRevisions });
        const group = groupedModels[0].group as DatasetGroup;

        expect(groupedModels).toHaveLength(1);
        expect(group.name).toBe('My Custom Dataset');

        expect(group.imageCount).toBe(100);
        expect(group.trainingSubsets).toEqual({
            training: 70,
            validation: 20,
            testing: 10,
        });
    });

    it('should fallback to dataset ID when revision not found', () => {
        const models = [
            getMockedModel({
                id: 'model-1',
                training_info: {
                    status: 'successful',
                    dataset_revision_id: 'dataset-unknown',
                    label_schema_revision: {},
                },
            }),
        ];

        const datasetRevisions: DatasetRevision[] = [];

        const groupedModels = groupModelsByDataset(models, t, { datasetRevisions });
        const group: DatasetGroup = groupedModels[0].group as DatasetGroup;

        expect(groupedModels).toHaveLength(1);
        expect(group.name).toBe('Dataset #dataset-');

        expect(group.imageCount).toBe(0);
        expect(group.trainingSubsets).toEqual({
            training: 0,
            validation: 0,
            testing: 0,
        });
    });
});

describe('groupModelsByArchitecture', () => {
    it('should return empty array for empty models', () => {
        const groupedModels = groupModelsByArchitecture([]);

        expect(groupedModels).toEqual([]);
    });

    it('should group models by architecture', () => {
        const models = [
            getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
            getMockedModel({ id: 'model-2', architecture: 'YOLOX' }),
            getMockedModel({ id: 'model-3', architecture: 'MobileNet' }),
        ];

        const groupedModels = groupModelsByArchitecture(models);

        expect(groupedModels).toHaveLength(2);

        const yoloxGroup = groupedModels.find(({ group }) => group.id === 'YOLOX');
        const mobileNetGroup = groupedModels.find(({ group }) => group.id === 'MobileNet');

        expect(yoloxGroup?.models).toHaveLength(2);
        expect(mobileNetGroup?.models).toHaveLength(1);
    });
});
