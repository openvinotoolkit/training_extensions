// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';

import { sortGroupedModelsByDatasetRevisionDate, sortModels } from './sorting';

describe('sortModels', () => {
    describe('name', () => {
        it('returns empty array for empty models', () => {
            const result = sortModels([], 'name', []);

            expect(result).toEqual([]);
        });

        it('sorts models by name ascending', () => {
            const models = [
                getMockedModel({ id: 'charlie', name: 'Charlie' }),
                getMockedModel({ id: 'alpha', name: 'Alpha' }),
                getMockedModel({ id: 'bravo', name: 'Bravo' }),
            ];

            const sorted = sortModels(models, 'name', []);

            expect(sorted[0].id).toBe('alpha');
            expect(sorted[1].id).toBe('bravo');
            expect(sorted[2].id).toBe('charlie');
        });

        it('sorts models by name case-insensitively', () => {
            const models = [
                getMockedModel({ id: 'upper', name: 'ZEBRA' }),
                getMockedModel({ id: 'lower', name: 'apple' }),
            ];

            const sorted = sortModels(models, 'name', []);

            expect(sorted[0].id).toBe('lower');
            expect(sorted[1].id).toBe('upper');
        });
    });

    describe('architecture', () => {
        it('sorts models by architecture ascending', () => {
            const models = [
                getMockedModel({ id: 'model-1', architecture: 'YOLOX' }),
                getMockedModel({ id: 'model-2', architecture: 'MobileNet' }),
                getMockedModel({ id: 'model-3', architecture: 'ResNet' }),
            ];

            const sorted = sortModels(models, 'architecture', []);

            expect(sorted[0].architecture).toBe('MobileNet');
            expect(sorted[1].architecture).toBe('ResNet');
            expect(sorted[2].architecture).toBe('YOLOX');
        });

        it('sorts models by architecture case-insensitively', () => {
            const models = [
                getMockedModel({ id: 'upper', architecture: 'YOLOX' }),
                getMockedModel({ id: 'lower', architecture: 'alexnet' }),
            ];

            const sorted = sortModels(models, 'architecture', []);

            expect(sorted[0].id).toBe('lower');
            expect(sorted[1].id).toBe('upper');
        });
    });

    describe('trained', () => {
        it('sorts models by training end time descending (most recent first)', () => {
            const models = [
                getMockedModel({
                    id: 'oldest',
                    training_info: { status: 'successful', end_time: '2025-01-01T00:00:00Z' },
                }),
                getMockedModel({
                    id: 'newest',
                    training_info: { status: 'successful', end_time: '2025-03-01T00:00:00Z' },
                }),
                getMockedModel({
                    id: 'middle',
                    training_info: { status: 'successful', end_time: '2025-02-01T00:00:00Z' },
                }),
            ];

            const sorted = sortModels(models, 'trained', []);

            expect(sorted[0].id).toBe('newest');
            expect(sorted[1].id).toBe('middle');
            expect(sorted[2].id).toBe('oldest');
        });

        it('places models with no end_time last (value 0)', () => {
            const models = [
                getMockedModel({ id: 'no-date', training_info: { status: 'not_started', end_time: null } }),
                getMockedModel({
                    id: 'has-date',
                    training_info: { status: 'successful', end_time: '2025-01-01T00:00:00Z' },
                }),
            ];

            const sorted = sortModels(models, 'trained', []);

            expect(sorted[0].id).toBe('has-date');
            expect(sorted[1].id).toBe('no-date');
        });

        it('places models with invalid end_time last (value 0)', () => {
            const models = [
                getMockedModel({
                    id: 'invalid-date',
                    training_info: { status: 'not_started', end_time: 'not-a-date' },
                }),
                getMockedModel({
                    id: 'valid-date',
                    training_info: { status: 'successful', end_time: '2025-06-01T00:00:00Z' },
                }),
            ];

            const sorted = sortModels(models, 'trained', []);

            expect(sorted[0].id).toBe('valid-date');
            expect(sorted[1].id).toBe('invalid-date');
        });
    });

    describe('device', () => {
        const modelsWithDevice = [
            getMockedModel({
                id: 'gpu',
                training_info: { status: 'successful', device: { type: 'cuda', name: 'NVIDIA RTX 3090' } },
            }),
            getMockedModel({
                id: 'cpu',
                training_info: { status: 'successful', device: { type: 'cpu', name: 'Intel Core Ultra 9' } },
            }),
            getMockedModel({
                id: 'xpu',
                training_info: { status: 'successful', device: { type: 'xpu', name: 'Intel Arc A770' } },
            }),
        ];

        it('sorts models by device name ascending', () => {
            const sorted = sortModels(modelsWithDevice, 'device', []);

            expect(sorted[0].id).toBe('xpu');
            expect(sorted[1].id).toBe('cpu');
            expect(sorted[2].id).toBe('gpu');
        });

        it('sorts models by device name case-insensitively', () => {
            const models = [
                getMockedModel({
                    id: 'cuda',
                    training_info: { status: 'successful', device: { type: 'cuda', name: 'ZOTAC GPU' } },
                }),
                getMockedModel({
                    id: 'cpu',
                    training_info: { status: 'successful', device: { type: 'cpu', name: 'amd cpu' } },
                }),
            ];

            const sorted = sortModels(models, 'device', []);

            expect(sorted[0].id).toBe('cpu');
            expect(sorted[1].id).toBe('cuda');
        });

        it('places models with an undefined device last', () => {
            const models = [
                getMockedModel({
                    id: 'undefined-device',
                    training_info: { status: 'successful', device: undefined },
                }),
                getMockedModel({
                    id: 'has-device',
                    training_info: { status: 'successful', device: { type: 'cpu', name: 'Intel Core Ultra 9' } },
                }),
            ];

            const sorted = sortModels(models, 'device', []);

            expect(sorted[0].id).toBe('has-device');
            expect(sorted[1].id).toBe('undefined-device');
        });
    });

    describe('size', () => {
        it('sorts models by size ascending', () => {
            const models = [
                getMockedModel({ id: 'large', size: 3000000 }),
                getMockedModel({ id: 'small', size: 500000 }),
                getMockedModel({ id: 'medium', size: 1500000 }),
            ];

            const sorted = sortModels(models, 'size', []);

            expect(sorted[0].id).toBe('small');
            expect(sorted[1].id).toBe('medium');
            expect(sorted[2].id).toBe('large');
        });
    });

    describe('score', () => {
        const makeModelWithScore = (id: string, primaryValue: number) =>
            getMockedModel({
                id,
                variants: [
                    getMockedVariant({
                        evaluations: [
                            {
                                dataset_revision_id: 'rev-1',
                                subset: 'testing',
                                metrics: [{ name: 'Accuracy', value: primaryValue, primary: true }],
                            },
                        ],
                    }),
                ],
            });

        it('sorts models by primary testing metric value descending', () => {
            const models = [
                makeModelWithScore('high', 0.9),
                makeModelWithScore('low', 0.5),
                makeModelWithScore('mid', 0.7),
            ];

            const sorted = sortModels(models, 'score', []);

            expect(sorted[0].id).toBe('high');
            expect(sorted[1].id).toBe('mid');
            expect(sorted[2].id).toBe('low');
        });

        it('places models with no testing metric last (value 0)', () => {
            const models = [getMockedModel({ id: 'no-metric', variants: [] }), makeModelWithScore('has-metric', 0.8)];

            const sorted = sortModels(models, 'score', []);

            expect(sorted[0].id).toBe('has-metric');
            expect(sorted[1].id).toBe('no-metric');
        });
    });

    describe('dataset', () => {
        it('sorts models by dataset revision creation date, newest first', () => {
            const revisions = [
                getMockedDatasetRevision({ id: 'rev-old', name: 'Alpha Dataset', created_at: '2025-01-01T00:00:00Z' }),
                getMockedDatasetRevision({ id: 'rev-new', name: 'Bravo Dataset', created_at: '2025-03-01T00:00:00Z' }),
                getMockedDatasetRevision({
                    id: 'rev-mid',
                    name: 'Charlie Dataset',
                    created_at: '2025-02-01T00:00:00Z',
                }),
            ];
            const models = [
                getMockedModel({
                    id: 'model-old',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-old' },
                }),
                getMockedModel({
                    id: 'model-new',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-new' },
                }),
                getMockedModel({
                    id: 'model-mid',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-mid' },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            expect(sorted[0].id).toBe('model-new');
            expect(sorted[1].id).toBe('model-mid');
            expect(sorted[2].id).toBe('model-old');
        });

        it('sorts by name Z->A when creation dates are equal', () => {
            const sharedDate = '2025-06-01T00:00:00Z';
            const revisions = [
                getMockedDatasetRevision({ id: 'rev-a', name: 'Alpha Dataset', created_at: sharedDate }),
                getMockedDatasetRevision({ id: 'rev-c', name: 'Charlie Dataset', created_at: sharedDate }),
                getMockedDatasetRevision({ id: 'rev-b', name: 'Bravo Dataset', created_at: sharedDate }),
            ];
            const models = [
                getMockedModel({
                    id: 'model-a',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-a' },
                }),
                getMockedModel({
                    id: 'model-c',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-c' },
                }),
                getMockedModel({
                    id: 'model-b',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-b' },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            expect(sorted[0].id).toBe('model-c');
            expect(sorted[1].id).toBe('model-b');
            expect(sorted[2].id).toBe('model-a');
        });

        it('places models with no dataset_revision_id last', () => {
            const revisions = [getMockedDatasetRevision({ id: 'rev-1', name: 'Zebra Dataset' })];
            const models = [
                getMockedModel({
                    id: 'no-dataset',
                    training_info: { status: 'not_started', dataset_revision_id: null },
                }),
                getMockedModel({
                    id: 'has-dataset',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-1' },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            expect(sorted[0].id).toBe('has-dataset');
            expect(sorted[1].id).toBe('no-dataset');
        });

        it('places models with a dataset_revision_id not found in revisions last', () => {
            const revisions = [getMockedDatasetRevision({ id: 'rev-known', name: 'Known Dataset' })];
            const models = [
                getMockedModel({
                    id: 'unknown',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-unknown' },
                }),
                getMockedModel({
                    id: 'known',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-known' },
                }),
            ];

            const sorted = sortModels(models, 'dataset', revisions);

            expect(sorted[0].id).toBe('known');
            expect(sorted[1].id).toBe('unknown');
        });

        it('returns empty array when given no models', () => {
            const revisions = [getMockedDatasetRevision()];

            expect(sortModels([], 'dataset', revisions)).toEqual([]);
        });
    });

    describe('default (unknown sort)', () => {
        it('returns the original models array and logs an error for an unknown sort option', () => {
            const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);

            const models = [getMockedModel({ id: 'a' }), getMockedModel({ id: 'b' })];
            // Cast to bypass TypeScript's exhaustive check — tests the runtime default branch
            const result = sortModels(models, 'unknown' as never, []);

            expect(result).toBe(models);
            expect(errorSpy).toHaveBeenCalledWith('Unknown sort option: unknown');

            errorSpy.mockRestore();
        });
    });

    describe('explicit direction', () => {
        it('reverses the name order', () => {
            const models = [
                getMockedModel({ id: 'alpha', name: 'Alpha' }),
                getMockedModel({ id: 'zulu', name: 'Zulu' }),
            ];

            expect(sortModels(models, 'name', [], 'desc').map(({ id }) => id)).toEqual(['zulu', 'alpha']);
        });

        it('reverses the size order', () => {
            const models = [getMockedModel({ id: 'small', size: 1 }), getMockedModel({ id: 'large', size: 2 })];

            expect(sortModels(models, 'size', [], 'desc').map(({ id }) => id)).toEqual(['large', 'small']);
        });

        it('reverses the training date order', () => {
            const models = [
                getMockedModel({
                    id: 'newest',
                    training_info: { status: 'successful', end_time: '2025-03-01T00:00:00Z' },
                }),
                getMockedModel({
                    id: 'oldest',
                    training_info: { status: 'successful', end_time: '2025-01-01T00:00:00Z' },
                }),
            ];

            expect(sortModels(models, 'trained', [], 'asc').map(({ id }) => id)).toEqual(['oldest', 'newest']);
        });

        it('keeps models without a training date last in both directions', () => {
            const models = [
                getMockedModel({ id: 'no-date', training_info: { status: 'not_started', end_time: null } }),
                getMockedModel({
                    id: 'has-date',
                    training_info: { status: 'successful', end_time: '2025-01-01T00:00:00Z' },
                }),
            ];

            expect(sortModels(models, 'trained', [], 'asc').map(({ id }) => id)).toEqual(['has-date', 'no-date']);
            expect(sortModels(models, 'trained', [], 'desc').map(({ id }) => id)).toEqual(['has-date', 'no-date']);
        });

        it('keeps models without a device last in both directions', () => {
            const models = [
                getMockedModel({ id: 'no-device', training_info: { status: 'successful', device: undefined } }),
                getMockedModel({
                    id: 'has-device',
                    training_info: { status: 'successful', device: { type: 'cpu', name: 'Intel Core Ultra 9' } },
                }),
            ];

            expect(sortModels(models, 'device', [], 'asc').map(({ id }) => id)).toEqual(['has-device', 'no-device']);
            expect(sortModels(models, 'device', [], 'desc').map(({ id }) => id)).toEqual(['has-device', 'no-device']);
        });

        it('keeps models without a score last in both directions', () => {
            const models = [
                getMockedModel({ id: 'no-metric', variants: [] }),
                getMockedModel({
                    id: 'has-metric',
                    variants: [
                        getMockedVariant({
                            evaluations: [
                                {
                                    dataset_revision_id: 'rev-1',
                                    subset: 'testing',
                                    metrics: [{ name: 'Accuracy', value: 0.8, primary: true }],
                                },
                            ],
                        }),
                    ],
                }),
            ];

            expect(sortModels(models, 'score', [], 'asc').map(({ id }) => id)).toEqual(['has-metric', 'no-metric']);
            expect(sortModels(models, 'score', [], 'desc').map(({ id }) => id)).toEqual(['has-metric', 'no-metric']);
        });

        it('keeps models without a dataset revision last in both directions', () => {
            const revisions = [getMockedDatasetRevision({ id: 'rev-1', name: 'Zebra Dataset' })];
            const models = [
                getMockedModel({
                    id: 'no-dataset',
                    training_info: { status: 'not_started', dataset_revision_id: null },
                }),
                getMockedModel({
                    id: 'has-dataset',
                    training_info: { status: 'successful', dataset_revision_id: 'rev-1' },
                }),
            ];

            expect(sortModels(models, 'dataset', revisions, 'asc').map(({ id }) => id)).toEqual([
                'has-dataset',
                'no-dataset',
            ]);
            expect(sortModels(models, 'dataset', revisions, 'desc').map(({ id }) => id)).toEqual([
                'has-dataset',
                'no-dataset',
            ]);
        });
    });
});

describe('sortGroupedModelsByDatasetRevisionDate', () => {
    // Builds a minimal DatasetGroup; the createdAt display string is irrelevant to ordering.
    const makeDatasetGroup = (id: string, models: ReturnType<typeof getMockedModel>[]) => ({
        group: {
            type: 'dataset' as const,
            id,
            name: `Dataset ${id}`,
            createdAt: '-',
            labelCount: 0,
            imageCount: 0,
            trainingSubsets: { training: 0, validation: 0, testing: 0 },
            filesDeleted: false,
        },
        models,
    });

    const makeModel = (id: string, revisionId: string) =>
        getMockedModel({ id, training_info: { status: 'successful', dataset_revision_id: revisionId } });

    it('returns empty array for empty input', () => {
        expect(sortGroupedModelsByDatasetRevisionDate([], [])).toEqual([]);
    });

    it('returns a single group unchanged', () => {
        const group = makeDatasetGroup('rev-only', []);
        const result = sortGroupedModelsByDatasetRevisionDate([group], []);

        expect(result).toHaveLength(1);
        expect(result[0].group.id).toBe('rev-only');
    });

    it('places the group with the newest dataset revision first, oldest last', () => {
        // Jan < Feb < Mar chronologically. The correct desc order is Mar, Feb, Jan.
        const revisions = [
            getMockedDatasetRevision({ id: 'rev-jan', created_at: '2025-01-01T00:00:00Z' }),
            getMockedDatasetRevision({ id: 'rev-feb', created_at: '2025-02-01T00:00:00Z' }),
            getMockedDatasetRevision({ id: 'rev-mar', created_at: '2025-03-01T00:00:00Z' }),
        ];
        const groups = [
            makeDatasetGroup('group-jan', [makeModel('model-jan', 'rev-jan')]),
            makeDatasetGroup('group-feb', [makeModel('model-feb', 'rev-feb')]),
            makeDatasetGroup('group-mar', [makeModel('model-mar', 'rev-mar')]),
        ];

        const result = sortGroupedModelsByDatasetRevisionDate(groups, revisions);

        expect(result[0].group.id).toBe('group-mar');
        expect(result[1].group.id).toBe('group-feb');
        expect(result[2].group.id).toBe('group-jan');
    });

    it('newly trained model group appears at the top even when given last in input order', () => {
        // Regression guard for the original bug: the group with the most recent revision
        // must always be rendered first regardless of the order returned by the API.
        const revisions = [
            getMockedDatasetRevision({ id: 'rev-old', created_at: '2025-01-01T00:00:00Z' }),
            getMockedDatasetRevision({ id: 'rev-new', created_at: '2025-06-01T00:00:00Z' }),
        ];
        const olderGroup = makeDatasetGroup('group-old', [makeModel('model-old', 'rev-old')]);
        const newerGroup = makeDatasetGroup('group-new', [makeModel('model-new', 'rev-new')]);

        // Intentionally supply the newer group last to confirm input order has no influence.
        const result = sortGroupedModelsByDatasetRevisionDate([olderGroup, newerGroup], revisions);

        expect(result[0].group.id).toBe('group-new');
        expect(result[1].group.id).toBe('group-old');
    });

    it('group order is driven by revision date, not by number of models per group', () => {
        // A group with many models on an older revision must not outrank a group
        // with fewer models on a newer revision.
        const revisions = [
            getMockedDatasetRevision({ id: 'rev-old', created_at: '2025-01-01T00:00:00Z' }),
            getMockedDatasetRevision({ id: 'rev-new', created_at: '2025-06-01T00:00:00Z' }),
        ];
        const bigOlderGroup = makeDatasetGroup('group-old', [
            makeModel('a1', 'rev-old'),
            makeModel('a2', 'rev-old'),
            makeModel('a3', 'rev-old'),
        ]);
        const smallNewerGroup = makeDatasetGroup('group-new', [makeModel('b1', 'rev-new')]);

        const result = sortGroupedModelsByDatasetRevisionDate([bigOlderGroup, smallNewerGroup], revisions);

        expect(result[0].group.id).toBe('group-new');
        expect(result[1].group.id).toBe('group-old');
    });

    it('group whose dataset revision is not in the revisions list sinks to the bottom', () => {
        // If the API returns a model whose dataset_revision_id has no matching revision,
        // that group must not float to the top and obscure more recent, known groups.
        const revisions = [getMockedDatasetRevision({ id: 'rev-known', created_at: '2025-01-01T00:00:00Z' })];
        const missingRevGroup = makeDatasetGroup('group-missing', [makeModel('model-missing', 'rev-unknown')]);
        const knownRevGroup = makeDatasetGroup('group-known', [makeModel('model-known', 'rev-known')]);

        const result = sortGroupedModelsByDatasetRevisionDate([missingRevGroup, knownRevGroup], revisions);

        expect(result[0].group.id).toBe('group-known');
        expect(result[1].group.id).toBe('group-missing');
    });
});
