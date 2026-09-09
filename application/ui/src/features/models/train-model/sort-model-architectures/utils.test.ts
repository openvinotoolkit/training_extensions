// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModelArchitecture } from 'mocks/mock-model';

import { SORTING_HANDLERS, SortingOptions } from './utils';

describe('SORTING_HANDLERS', () => {
    describe('edge cases', () => {
        it('returns an empty array unchanged for any handler', () => {
            for (const handler of Object.values(SORTING_HANDLERS)) {
                expect(handler([])).toEqual([]);
            }
        });

        it('returns a single-element array unchanged for any handler', () => {
            const single = [getMockedModelArchitecture()];

            for (const handler of Object.values(SORTING_HANDLERS)) {
                expect(handler(single)).toEqual(single);
            }
        });
    });

    describe('Sort by name', () => {
        const architectures = [
            getMockedModelArchitecture({ name: 'Zebra' }),
            getMockedModelArchitecture({ name: 'Alpha' }),
            getMockedModelArchitecture({ name: 'Mango' }),
        ];
        it('sorts alphabetically A to Z by name', () => {
            const result = SORTING_HANDLERS[SortingOptions.NAME_ASC](architectures);

            expect(result.map((a) => a.name)).toEqual(['Alpha', 'Mango', 'Zebra']);
        });

        it('sorts alphabetically Z to A by name', () => {
            const result = SORTING_HANDLERS[SortingOptions.NAME_DESC](architectures);

            expect(result.map((a) => a.name)).toEqual(['Zebra', 'Mango', 'Alpha']);
        });
    });

    describe('Sort by speed', () => {
        const architectures = [
            getMockedModelArchitecture({
                stats: { gigaflops: 200, trainable_parameters: 10, benchmark_metrics: {} },
            }),
            getMockedModelArchitecture({
                stats: { gigaflops: 50, trainable_parameters: 10, benchmark_metrics: {} },
            }),
            getMockedModelArchitecture({
                stats: { gigaflops: 120, trainable_parameters: 10, benchmark_metrics: {} },
            }),
        ];
        it('sorts ascending by gigaflops (fastest = lowest GFLOPs first)', () => {
            const result = SORTING_HANDLERS[SortingOptions.SPEED_ASC](architectures);

            expect(result.map((a) => a.stats?.gigaflops)).toEqual([50, 120, 200]);
        });

        it('sorts descending by gigaflops (slowest = highest GFLOPs first)', () => {
            const result = SORTING_HANDLERS[SortingOptions.SPEED_DESC](architectures);

            expect(result.map((a) => a.stats?.gigaflops)).toEqual([200, 120, 50]);
        });
    });

    describe('Sort by accuracy ascending', () => {
        it('sorts by imagenet_top1_accuracy when present', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'High',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 90 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Low',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 30 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Mid',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 60 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_ASC](architectures);

            expect(result.map((a) => a.name)).toEqual(['Low', 'Mid', 'High']);
        });

        it('falls back to coco_map_50_95 when imagenet_top1_accuracy is null', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'High',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: 0.8 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Low',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: 0.3 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Mid',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: 0.5 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_ASC](architectures);

            expect(result.map((a) => a.name)).toEqual(['Low', 'Mid', 'High']);
        });

        it('falls back to coco_map_50 when both imagenet_top1_accuracy and coco_map_50_95 are null', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'High',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: null, coco_map_50: 0.9 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Low',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: null, coco_map_50: 0.2 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Mid',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: null, coco_map_50: 0.55 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_ASC](architectures);

            expect(result.map((a) => a.name)).toEqual(['Low', 'Mid', 'High']);
        });

        it('uses imagenet_top1_accuracy over coco_map_50_95 when both are present', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'A',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 10, coco_map_50_95: 90 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'B',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 80, coco_map_50_95: 5 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_ASC](architectures);

            expect(result.map((a) => a.name)).toEqual(['A', 'B']);
        });
    });

    describe('Sorts by accuracy descending', () => {
        it('sorts by imagenet_top1_accuracy when present', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'Low',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 30 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'High',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 90 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Mid',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: 60 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_DESC](architectures);

            expect(result.map((a) => a.name)).toEqual(['High', 'Mid', 'Low']);
        });

        it('falls back to coco_map_50_95 when imagenet_top1_accuracy is null', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'Mid',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: 0.5 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'High',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: 0.8 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Low',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: 0.3 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_DESC](architectures);

            expect(result.map((a) => a.name)).toEqual(['High', 'Mid', 'Low']);
        });

        it('falls back to coco_map_50 when both imagenet_top1_accuracy and coco_map_50_95 are null', () => {
            const architectures = [
                getMockedModelArchitecture({
                    name: 'Mid',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: null, coco_map_50: 0.55 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'High',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: null, coco_map_50: 0.9 },
                    },
                }),
                getMockedModelArchitecture({
                    name: 'Low',
                    stats: {
                        gigaflops: 10,
                        trainable_parameters: 10,
                        benchmark_metrics: { imagenet_top1_accuracy: null, coco_map_50_95: null, coco_map_50: 0.2 },
                    },
                }),
            ];

            const result = SORTING_HANDLERS[SortingOptions.ACCURACY_DESC](architectures);

            expect(result.map((a) => a.name)).toEqual(['High', 'Mid', 'Low']);
        });
    });
});
