// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';
import { renderHook } from 'test-utils/render';

import { isFailedModel } from '../utils/utils';
import { useGroupedModels } from './use-grouped-models.hook';

describe('useGroupedModels', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('basic functionality', () => {
        it('should return empty array when models is undefined', () => {
            const { result } = renderHook(() =>
                useGroupedModels(undefined, {
                    groupBy: 'dataset',
                    sortBy: {},
                    searchBy: '',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            expect(result.current).toEqual([]);
        });

        it('should return empty array when models is empty', () => {
            const { result } = renderHook(() =>
                useGroupedModels([], {
                    groupBy: 'dataset',
                    sortBy: {},
                    searchBy: '',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            expect(result.current).toEqual([]);
        });
    });

    describe('grouping by dataset', () => {
        it('should group models by dataset', () => {
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

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'dataset',
                    sortBy: {},
                    searchBy: '',
                    datasetRevisions: [
                        getMockedDatasetRevision({ id: 'dataset-1' }),
                        getMockedDatasetRevision({ id: 'dataset-2' }),
                    ],
                    showFailedModels: true,
                })
            );

            expect(result.current).toHaveLength(2);
            expect(result.current[0].models).toHaveLength(2);
            expect(result.current[1].models).toHaveLength(1);
        });
    });

    describe('grouping by architecture', () => {
        it('should group models by architecture', () => {
            const models = [
                getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', architecture: 'MobileNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: '',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            expect(result.current).toHaveLength(2);

            const yoloxGroup = result.current.find(({ group }) => group.id === 'YOLOX');
            const mobileNetGroup = result.current.find(({ group }) => group.id === 'MobileNet');

            expect(yoloxGroup?.models).toHaveLength(2);
            expect(mobileNetGroup?.models).toHaveLength(1);
        });
    });

    describe('search filtering', () => {
        it('should return all models when searchBy is empty', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S' }),
                getMockedModel({ id: 'model-3', name: 'MobileNet-V2' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: '',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            const allModels = result.current.flatMap((group) => group.models);
            expect(allModels).toHaveLength(3);
        });

        it('should filter models by name (case-insensitive)', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'resnet-101', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: 'resnet',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            const allModels = result.current.flatMap((group) => group.models);
            expect(allModels).toHaveLength(2);
            expect(allModels.map((m) => m.name)).toEqual(expect.arrayContaining(['ResNet-50', 'resnet-101']));
        });

        it('should return empty groups when no models match search query', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: 'nonexistent',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            expect(result.current).toHaveLength(0);
        });

        it('should filter out empty groups after search', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'ResNet-50', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'YOLOX-S', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'ResNet-101', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: 'YOLOX',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            // Only YOLOX group should remain, ResNet group should be filtered out
            expect(result.current).toHaveLength(1);
            expect(result.current[0].group.id).toBe('YOLOX');
        });

        it('should apply search filter before grouping and sorting', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'Alpha-ResNet', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'Beta-YOLOX', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-3', name: 'Gamma-ResNet', architecture: 'ResNet' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: { ResNet: { key: 'name', direction: 'asc' } },
                    searchBy: 'ResNet',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            expect(result.current).toHaveLength(1);
            expect(result.current[0].models).toHaveLength(2);

            expect(result.current[0].models[0].name).toBe('Alpha-ResNet');
            expect(result.current[0].models[1].name).toBe('Gamma-ResNet');
        });

        it('should match partial names', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'My-Custom-Model-v1' }),
                getMockedModel({ id: 'model-2', name: 'Another-Model' }),
                getMockedModel({ id: 'model-3', name: 'Custom-Detection' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: 'Custom',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            const allModels = result.current.flatMap((group) => group.models);
            expect(allModels).toHaveLength(2);
            expect(allModels.map((m) => m.name)).toEqual(
                expect.arrayContaining(['My-Custom-Model-v1', 'Custom-Detection'])
            );
        });

        it('filters out failed models', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'My-Custom-Model-v1' }),
                getMockedModel({ id: 'model-2', name: 'Another-Model' }),
                getMockedModel({ id: 'model-3', name: 'Custom-Detection', training_info: { status: 'failed' } }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {},
                    searchBy: '',
                    datasetRevisions: [],
                    showFailedModels: false,
                })
            );

            const allModels = result.current.flatMap((group) => group.models);
            const notFailedModels = allModels.filter((model) => !isFailedModel(model));

            expect(allModels).toHaveLength(notFailedModels.length);
            expect(notFailedModels.map((m) => m.name)).toEqual(
                expect.arrayContaining(['My-Custom-Model-v1', 'Another-Model'])
            );
        });
    });

    describe('sorting', () => {
        it('sorts each group independently', () => {
            const models = [
                getMockedModel({ id: 'model-1', name: 'Alpha', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-2', name: 'Beta', architecture: 'ResNet' }),
                getMockedModel({ id: 'model-3', name: 'Gamma', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-4', name: 'Delta', architecture: 'YOLOX' }),
            ];

            const { result } = renderHook(() =>
                useGroupedModels(models, {
                    groupBy: 'architecture',
                    sortBy: {
                        ResNet: { key: 'name', direction: 'asc' },
                        YOLOX: { key: 'name', direction: 'desc' },
                    },
                    searchBy: '',
                    datasetRevisions: [],
                    showFailedModels: true,
                })
            );

            const groupNames = (id: string) =>
                result.current.find(({ group }) => group.id === id)?.models.map(({ name }) => name);

            expect(groupNames('ResNet')).toEqual(['Alpha', 'Beta']);
            expect(groupNames('YOLOX')).toEqual(['Gamma', 'Delta']);
        });
    });
});
