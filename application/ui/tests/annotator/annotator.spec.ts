// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO, DatasetSubset, PredictionDTO } from '@/api/types';
import { expect, type Locator } from '@playwright/test';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { FEATURE_FLAGS } from '../../src/constants/feature-flags';
import { Polygon } from '../../src/shared/types';
import { http, test } from '../fixtures';
import { blueLabel, candyBinaryHandler, redLabel } from './annotator-fixtures';

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'detection',
        labels: [redLabel, blueLabel],
    },
});

test.describe('Annotator', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            http.get('/api/projects', () => {
                return HttpResponse.json([mockedDetectionProject]);
            }),
            candyBinaryHandler,
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );
    });

    test('Add and change annotations labels', async ({ page, boundingBoxTool, annotatorPage }) => {
        await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

        await test.step('Draw an annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });

            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
        });

        await test.step('Change annotation label by clicking label badge', async () => {
            await page.getByRole('button', { name: 'Selection' }).click();
            await page.getByLabel('annotation rect').nth(1).click();

            await expect(page.getByRole('button', { name: `Label ${redLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );

            await page.getByRole('button', { name: `Label ${blueLabel.name}` }).click();

            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(1);
            await expect(page.getByRole('button', { name: `Label ${blueLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );
        });

        await test.step('Draw a second annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 300, y: 200, width: 150, height: 150 });

            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(2);
        });

        await test.step('Change second annotation to red label', async () => {
            await page.getByRole('button', { name: 'Selection' }).click();
            await page.getByLabel('annotation rect').nth(3).click();
            await page.getByRole('button', { name: `Label ${redLabel.name}` }).click();

            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
        });

        await test.step('Verify both annotations have correct labels', async () => {
            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(1);
            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
        });
    });

    test('change multiple labels at once', async ({ page, boundingBoxTool, annotatorPage }) => {
        await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

        const annotations = [
            { x: 100, y: 100, width: 150, height: 150 },
            { x: 300, y: 200, width: 150, height: 150 },
            { x: 600, y: 300, width: 150, height: 150 },
        ];

        await test.step('Draw annotations', async () => {
            await boundingBoxTool.selectTool();

            for await (const annotation of annotations) {
                await boundingBoxTool.drawBoundingBox(annotation);
            }

            await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(annotations.length);
        });

        await test.step('Remove labels', async () => {
            await page.getByRole('button', { name: 'Selection' }).click();
            const labels = page.getByLabel('Remove red-label');

            await labels.nth(0).click();
            await labels.nth(1).click();
        });

        await test.step('Change selected annotations label using label badge', async () => {
            const container = page.getByLabel('annotation rect');

            await container.nth(5).click();
            await container.nth(4).click({ modifiers: ['Shift'] });
            await container.nth(3).click({ modifiers: ['Shift'] });

            await page.getByRole('button', { name: `Label ${blueLabel.name}` }).click();

            await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(annotations.length);
        });
    });

    test.describe('Handles empty label', () => {
        test('label assignment', async ({ page, boundingBoxTool, annotatorPage }) => {
            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            const annotations = [
                { x: 100, y: 100, width: 150, height: 150 },
                { x: 300, y: 200, width: 150, height: 150 },
                { x: 600, y: 300, width: 150, height: 150 },
            ];

            await test.step('Draw annotations', async () => {
                await boundingBoxTool.selectTool();

                for await (const annotation of annotations) {
                    await boundingBoxTool.drawBoundingBox(annotation);
                }

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(annotations.length);
            });

            await test.step('Assigning "No object" removes other annotations', async () => {
                await page.getByLabel('Label No object').click();

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(0);
                await expect(page.getByLabel(`label No object background`)).toHaveCount(1);
            });

            await test.step('Drawing new annotation removes "No object" annotation', async () => {
                await boundingBoxTool.selectTool();

                await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
                await expect(page.getByLabel(`label No object background`)).toHaveCount(0);
            });
        });

        test('renders "No object" when server returns empty annotations list', async ({ page, annotatorPage }) => {
            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            await expect(page.getByLabel(`label No object background`)).toHaveCount(1);
        });

        test('drawing a new annotation removes global annotation', async ({ page, boundingBoxTool, annotatorPage }) => {
            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('Verify global "No object" annotation is visible initially', async () => {
                await expect(annotatorPage.getAnnotationsList()).toBeVisible();
                await expect(page.getByLabel('label No object background')).toHaveCount(1);
            });

            await test.step('Remove the label, annotation still exists', async () => {
                await page.getByRole('button', { name: 'Remove No object' }).click();
                await expect(page.getByLabel('label No label background')).toHaveCount(1);
            });

            await test.step('Draw a new annotation', async () => {
                await boundingBoxTool.selectTool();
                await boundingBoxTool.drawBoundingBox({ x: 220, y: 180, width: 180, height: 160 });
            });

            await test.step('Global annotation without the label is removed and new annotation is visible', async () => {
                await expect(page.getByLabel('label No label background')).toHaveCount(0);
                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);
                await expect(page.getByLabel(`label ${redLabel.name} background`).first()).toBeVisible();
            });
        });
    });

    test('Tool selection persists across media items', async ({ polygonTool, annotatorPage, network }) => {
        const smallPolygon: Polygon = {
            type: 'polygon',
            points: [
                { x: 100, y: 100 },
                { x: 150, y: 100 },
                { x: 150, y: 150 },
                { x: 100, y: 150 },
            ],
        };

        const mediaItems = [
            getMockedMediaImage({ id: 'media-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];
        const mockedSegmentationProject = getMockedProject({
            id: '123e4567-e89b-12d3-a456-426614174000',
            task: {
                exclusive_labels: true,
                task_type: 'instance_segmentation',
                labels: [redLabel, blueLabel],
            },
        });

        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedSegmentationProject);
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedSegmentationProject.id, 'media-1');

        await test.step('Select polygon tool on first media item', async () => {
            await polygonTool.selectPolygonTool();
        });

        await test.step('Navigate to second media item by clicking in sidebar', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
        });

        await test.step('Verify polygon tool persisted by drawing a polygon', async () => {
            // If polygon tool persisted, we should be able to draw immediately without reselecting
            await polygonTool.drawPolygon(smallPolygon);

            expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(1);
        });

        await test.step('Navigate back to first media item', async () => {
            await annotatorPage.selectMediaItem('item-1.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
        });

        await test.step('Verify polygon tool still works after navigating back', async () => {
            // Draw another polygon to verify tool is still active
            await polygonTool.drawPolygon(smallPolygon);

            expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(1);
        });
    });

    test('Annotations reset correctly when switching media items', async ({ annotatorPage, network }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-reset-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-reset-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];

        const mediaAnnotations: Record<string, AnnotationDTO[]> = {
            'media-reset-1': [
                {
                    shape: {
                        type: 'rectangle',
                        x: 80,
                        y: 120,
                        width: 140,
                        height: 120,
                    },
                    labels: [{ id: redLabel.id }],
                },
            ],
            'media-reset-2': [],
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ params }) => {
                return HttpResponse.json({
                    annotations: mediaAnnotations[params.media_id] ?? [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedDetectionProject.id, 'media-reset-1');

        await test.step('Check first media annotations', async () => {
            await expect(annotatorPage.getAnnotationsList()).toBeVisible();

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
        });

        await test.step('Switching to media 2 clears media 1 annotations', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
        });

        await test.step('Switching back restores media 1 annotations', async () => {
            await annotatorPage.selectMediaItem('item-1.jpg');

            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
        });
    });

    test('Selected annotations reset when switching media items', async ({ page, annotatorPage, network }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-selection-reset-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-selection-reset-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];

        const mediaAnnotations: Record<string, AnnotationDTO[]> = {
            'media-selection-reset-1': [
                {
                    shape: {
                        type: 'rectangle',
                        x: 80,
                        y: 120,
                        width: 140,
                        height: 120,
                    },
                    labels: [{ id: redLabel.id }],
                },
            ],
            'media-selection-reset-2': [],
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ params }) => {
                return HttpResponse.json({
                    annotations: mediaAnnotations[params.media_id] ?? [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedDetectionProject.id, 'media-selection-reset-1');

        await test.step('Select annotation on media 1', async () => {
            await page.getByRole('button', { name: 'Selection' }).click();
            await page.getByLabel('annotation rect').nth(1).click();

            const selectedAnnotations = annotatorPage.getAnnotationsList().getByLabel('selected annotation');
            await expect(selectedAnnotations).toHaveCount(1);
        });

        await test.step('Switch to media 2 and back to media 1 resets selection', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');
            await expect(annotatorPage.getAnnotationsList()).toBeVisible();
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);

            await annotatorPage.selectMediaItem('item-1.jpg');
            await expect(annotatorPage.getAnnotationsList()).toBeVisible();

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

            const selectedAnnotations = annotatorPage.getAnnotationsList().getByLabel('selected annotation');
            await expect(selectedAnnotations).toHaveCount(0);
        });
    });

    test('Selected label persists when switching media items', async ({ page, network, annotatorPage }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];

        network.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () => {
                return HttpResponse.json({
                    annotations: [],
                    user_reviewed: true,
                    subset: 'training',
                });
            })
        );

        await annotatorPage.goto(mockedDetectionProject.id, 'media-1');

        await test.step('Select non-default label on first media item', async () => {
            const blueLabelButton = page.getByRole('button', { name: `Label ${blueLabel.name}` });
            await blueLabelButton.click();

            await expect(blueLabelButton).toHaveAttribute('aria-pressed', 'true');
        });

        await test.step('Switching media keeps selected label active', async () => {
            await annotatorPage.selectMediaItem('item-2.jpg');

            await expect(page.getByRole('button', { name: `Label ${blueLabel.name}` })).toHaveAttribute(
                'aria-pressed',
                'true'
            );
        });
    });

    test.describe('Annotation and prediction modes', () => {
        const predictions = [
            {
                shape: {
                    type: 'rectangle',
                    x: 3,
                    y: 0,
                    width: 780,
                    height: 421,
                },
                labels: [{ id: blueLabel.id }],
                confidences: [0.9619140625],
            },
            {
                shape: {
                    type: 'rectangle',
                    x: 1007,
                    y: 624,
                    width: 909,
                    height: 456,
                },
                labels: [{ id: blueLabel.id }],
                confidences: [0.9599609375],
            },
            {
                shape: {
                    type: 'rectangle',
                    x: 291,
                    y: 0,
                    width: 1553,
                    height: 889,
                },
                labels: [{ id: blueLabel.id }],
                confidences: [0.904296875],
            },
        ] satisfies PredictionDTO[];

        test('Annotation vs Prediction', async ({ page, annotatorPage, boundingBoxTool, network }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([
                        getMockedModel({
                            variants: [getMockedVariant({})],
                        }),
                    ]);
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                    return HttpResponse.json({
                        annotations: [],
                        user_reviewed: true,
                        subset: 'training',
                    });
                }),
                http.post('/api/projects/{project_id}/dataset/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: {
                                    id: '123',
                                },
                                prediction: predictions,
                            },
                        ],
                    });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('Draws an annotation in annotation mode', async () => {
                await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'true');
                await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'false');

                const annotation = { x: 100, y: 100, width: 150, height: 150 };

                await boundingBoxTool.selectTool();
                await boundingBoxTool.drawBoundingBox(annotation);

                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(1);

                expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(0);
            });

            await test.step('Displays server prediction in prediction mode', async () => {
                await annotatorPage.openPredictionMode();

                await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');
                await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'false');

                await expect(annotatorPage.getPrimaryToolbar()).toBeHidden();
                await expect(page.getByLabel('Labels', { exact: true })).toBeHidden();

                await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(predictions.length);
                expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(predictions.length);

                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
                await expect(page.getByLabel(`label ${redLabel.name} background`)).toHaveCount(0);
            });

            await test.step('Hides predictions in annotation mode, restores annotation', async () => {
                await annotatorPage.openAnnotationMode();

                expect(await annotatorPage.getAnnotationsListItems('prediction rect')).toHaveLength(0);
                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
            });

            await test.step('Edits prediction by overwriting existing annotations with prediction in annotation mode', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.editPrediction();

                await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'true');
                await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'false');

                await expect(page.getByLabel(`label ${blueLabel.name} background`)).toHaveCount(predictions.length);
                await expect(page.getByLabel(`label ${redLabel.name} background`)).toBeHidden();
            });
        });

        test('Displays cues for annotator modes only for the first time for one media, resets for next media', async ({
            annotatorPage,
            page,
            network,
        }) => {
            // item-1: no annotations (404), has predictions
            // item-2: has 1 annotation, has predictions
            const mediaItems = [
                getMockedMediaImage({ id: 'item-1', name: 'item-1', width: 1920, height: 1080 }),
                getMockedMediaImage({ id: 'item-2', name: 'item-2', width: 1920, height: 1080 }),
            ];

            network.use(
                http.get('/api/projects/{project_id}/dataset/media', () => {
                    return HttpResponse.json({
                        items: mediaItems,
                        pagination: {
                            offset: 0,
                            limit: 10,
                            count: mediaItems.length,
                            total: mediaItems.length,
                        },
                    });
                }),
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([
                        getMockedModel({
                            variants: [getMockedVariant({})],
                        }),
                    ]);
                }),
                http.post('/api/projects/{project_id}/dataset/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: {
                                    id: '123',
                                },
                                prediction: predictions,
                            },
                        ],
                    });
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ params }) => {
                    if (params.media_id === mediaItems[0].id) {
                        return HttpResponse.json(
                            {
                                // @ts-expect-error We care only about mocking detail
                                detail: 'Media has not been annotated yet',
                            },
                            { status: 404 }
                        );
                    }

                    return HttpResponse.json({
                        annotations: [
                            {
                                shape: {
                                    type: 'rectangle',
                                    x: 1007,
                                    y: 624,
                                    width: 909,
                                    height: 456,
                                },
                                labels: [{ id: redLabel.id }],
                            },
                        ],
                        user_reviewed: true,
                        subset: 'training',
                    });
                })
            );

            await test.step('item-1 (annotation mode): prediction cue visible because there are no annotations but predictions exist', async () => {
                const predictResponsePromise = page.waitForResponse((res) => res.url().includes('media:predict'));

                await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

                await expect(annotatorPage.getAnnotationsList()).toBeVisible();
                await predictResponsePromise;

                await expect(page.getByLabel('Prediction available')).toBeVisible();
            });

            await test.step('item-1: switching annotation -> prediction dismisses prediction cue; switching back keeps it dismissed', async () => {
                await annotatorPage.openPredictionMode();

                await annotatorPage.openAnnotationMode();

                await expect(page.getByLabel('Prediction available')).toBeHidden();
            });

            await test.step('navigate to item-2 (in prediction mode): no annotation cue is shown', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.selectMediaItem('item-2');

                await expect(page.getByLabel('Annotation available')).toBeHidden();
            });

            await test.step('item-2: switching prediction -> annotation hides prediction cue', async () => {
                await annotatorPage.openAnnotationMode();

                await expect(page.getByLabel('Prediction available')).toBeHidden();
            });
        });

        test('Displays "No object" when media:predict returns empty predictions', async ({
            page,
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([
                        getMockedModel({
                            variants: [getMockedVariant({})],
                        }),
                    ]);
                }),
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async () => {
                    return HttpResponse.json(
                        {
                            // @ts-expect-error We care only about mocking detail
                            detail: 'Media has not been annotated yet',
                        },
                        { status: 404 }
                    );
                }),
                http.post('/api/projects/{project_id}/dataset/media:predict', async () => {
                    return HttpResponse.json({
                        predictions: [
                            {
                                media: { id: '123' },
                                prediction: [],
                            },
                        ],
                    });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await annotatorPage.openPredictionMode();

            await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');
            await expect(page.getByLabel('label No object background')).toHaveCount(1);
        });
    });

    test.describe('Edit mode', () => {
        const mockedSegmentationProject = getMockedProject({
            id: '123e4567-e89b-12d3-a456-426614174000',
            task: {
                exclusive_labels: true,
                task_type: 'instance_segmentation',
                labels: [redLabel, blueLabel],
            },
        });

        const smallPolygon = {
            type: 'polygon' as const,
            points: [
                { x: 100, y: 100 },
                { x: 250, y: 100 },
                { x: 250, y: 250 },
                { x: 100, y: 250 },
            ],
        };

        const secondPolygon = {
            type: 'polygon' as const,
            points: [
                { x: 400, y: 300 },
                { x: 550, y: 300 },
                { x: 550, y: 450 },
                { x: 400, y: 450 },
            ],
        };

        test('detection task — edits bounding box while bounding box tool stays active', async ({
            page,
            boundingBoxTool,
            annotatorPage,
        }) => {
            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            const selectedAnnotationRect = page.getByLabel('selected annotation').getByLabel('annotation rect').first();
            const getRectGeometry = async (rect: Locator) => {
                const [x, y, width, height] = await Promise.all([
                    rect.getAttribute('x'),
                    rect.getAttribute('y'),
                    rect.getAttribute('width'),
                    rect.getAttribute('height'),
                ]);

                return {
                    x,
                    y,
                    width,
                    height,
                };
            };

            await test.step('Draw a bounding box and keep drawing tool active', async () => {
                await boundingBoxTool.selectTool();
                await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });

                await expect(boundingBoxTool.getTool()).toHaveAttribute('aria-pressed', 'true');
                await expect(page.getByLabel(/^Edit bounding box points/)).toHaveCount(1);
            });

            const initialGeometry = await getRectGeometry(selectedAnnotationRect);

            await test.step('Resize the bounding box from south-east resize anchor', async () => {
                const southEastAnchor = page
                    .getByLabel('selected annotation')
                    .getByLabel('South east resize anchor')
                    .first();
                const anchorBox = await southEastAnchor.boundingBox();

                expect(anchorBox).not.toBeNull();

                if (anchorBox === null) {
                    return;
                }

                await page.mouse.move(anchorBox.x + anchorBox.width / 2, anchorBox.y + anchorBox.height / 2);
                await page.mouse.down();
                await page.mouse.move(anchorBox.x + anchorBox.width / 2 + 50, anchorBox.y + anchorBox.height / 2 + 40);
                await page.mouse.up();
            });

            await test.step('Bounding box geometry changes without switching tools', async () => {
                await expect
                    .poll(async () => {
                        return getRectGeometry(selectedAnnotationRect);
                    })
                    .not.toEqual(initialGeometry);

                await expect(page.getByLabel(/^Edit bounding box points/)).toHaveCount(1);
            });

            await test.step('Can draw another bounding box without switching to selection tool', async () => {
                await boundingBoxTool.drawBoundingBox({ x: 350, y: 250, width: 120, height: 120 });

                expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(2);
            });
        });

        test('instance segmentation task — edits polygon point while polygon tool stays active', async ({
            page,
            polygonTool,
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}', () => {
                    return HttpResponse.json(mockedSegmentationProject);
                })
            );

            await annotatorPage.goto(mockedSegmentationProject.id, 'item-1');

            await test.step('Draw a polygon and keep drawing tool active', async () => {
                await polygonTool.selectPolygonTool();
                await polygonTool.drawPolygon(smallPolygon);

                await expect(polygonTool.getTool()).toHaveAttribute('aria-pressed', 'true');
                await expect(page.locator('[id^="edit-polygon-points-"]')).toHaveCount(1);
            });

            const polygonAnnotation = page.getByLabel('selected annotation').getByLabel('annotation polygon').first();
            const initialPoints = (await polygonAnnotation.getAttribute('points')) ?? '';

            await test.step('Move polygon anchor while polygon tool is still selected', async () => {
                const polygonAnchor = page
                    .getByLabel('selected annotation')
                    .getByLabel('Resize polygon (250, 100) anchor')
                    .first();
                const anchorBox = await polygonAnchor.boundingBox();

                expect(anchorBox).not.toBeNull();

                if (anchorBox === null) {
                    return;
                }

                await page.mouse.move(anchorBox.x + anchorBox.width / 2, anchorBox.y + anchorBox.height / 2);
                await page.mouse.down();
                await page.mouse.move(anchorBox.x + anchorBox.width / 2 + 30, anchorBox.y + anchorBox.height / 2 + 20);
                await page.mouse.up();
            });

            await test.step('Polygon geometry changes without switching tools', async () => {
                await expect
                    .poll(async () => (await polygonAnnotation.getAttribute('points')) ?? '')
                    .not.toBe(initialPoints);
                await expect(page.locator('[id^="edit-polygon-points-"]')).toHaveCount(1);
            });

            await test.step('Can draw another polygon without switching to selection tool', async () => {
                await polygonTool.drawPolygon(secondPolygon);

                expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(2);
            });
        });

        test.describe('Edit mode deselection', () => {
            test('detection task — new shape enters edit mode and next shape replaces active edit selection', async ({
                page,
                boundingBoxTool,
                annotatorPage,
            }) => {
                await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

                await test.step('Draw first bounding box and verify it enters edit mode immediately', async () => {
                    await boundingBoxTool.selectTool();
                    await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });

                    await expect(page.getByLabel(/^Edit bounding box points/)).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                });

                await test.step('Draw second bounding box with the same tool', async () => {
                    await boundingBoxTool.selectTool();
                    await boundingBoxTool.drawBoundingBox({ x: 350, y: 250, width: 150, height: 150 });
                });

                await test.step('Only the newly created annotation remains in edit mode', async () => {
                    await expect(page.getByLabel(/^Edit bounding box points/)).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                    expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(2);
                });
            });

            test('detection task — selection tool edit mode is replaced by newly drawn shape', async ({
                page,
                boundingBoxTool,
                annotatorPage,
            }) => {
                await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

                await test.step('Draw first bounding box', async () => {
                    await boundingBoxTool.selectTool();
                    await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });
                });

                await test.step('Enter edit mode via selection tool', async () => {
                    await page.getByRole('button', { name: 'Selection' }).click();
                    await page.getByLabel('annotation rect').nth(1).click();

                    await expect(page.getByLabel(/^Edit bounding box points/)).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                });

                await test.step('Draw second bounding box', async () => {
                    await boundingBoxTool.selectTool();
                    await boundingBoxTool.drawBoundingBox({ x: 350, y: 250, width: 150, height: 150 });
                });

                await test.step('Previously selected annotation is deselected and new one is in edit mode', async () => {
                    await expect(page.getByLabel(/^Edit bounding box points/)).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                    expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(2);
                });
            });

            test('instance segmentation task — new shape enters edit mode and next shape replaces active edit selection', async ({
                page,
                polygonTool,
                annotatorPage,
                network,
            }) => {
                network.use(
                    http.get('/api/projects/{project_id}', () => {
                        return HttpResponse.json(mockedSegmentationProject);
                    })
                );

                await annotatorPage.goto(mockedSegmentationProject.id, 'item-1');

                await test.step('Draw first polygon and verify it enters edit mode immediately', async () => {
                    await polygonTool.selectPolygonTool();
                    await polygonTool.drawPolygon(smallPolygon);

                    await expect(page.locator('[id^="edit-polygon-points-"]')).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                });

                await test.step('Draw second polygon with the same tool', async () => {
                    await polygonTool.selectPolygonTool();
                    await polygonTool.drawPolygon(secondPolygon);
                });

                await test.step('Only the newly created annotation remains in edit mode', async () => {
                    await expect(page.locator('[id^="edit-polygon-points-"]')).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                    expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(2);
                });
            });

            test('instance segmentation task — selection tool edit mode is replaced by newly drawn shape', async ({
                page,
                polygonTool,
                annotatorPage,
                network,
            }) => {
                network.use(
                    http.get('/api/projects/{project_id}', () => {
                        return HttpResponse.json(mockedSegmentationProject);
                    })
                );

                await annotatorPage.goto(mockedSegmentationProject.id, 'item-1');

                await test.step('Draw first polygon', async () => {
                    await polygonTool.selectPolygonTool();
                    await polygonTool.drawPolygon(smallPolygon);
                });

                await test.step('Enter edit mode via selection tool', async () => {
                    await page.getByRole('button', { name: 'Selection' }).click();
                    await page.getByLabel('annotation polygon').nth(1).click();

                    await expect(page.locator('[id^="edit-polygon-points-"]')).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                });

                await test.step('Draw second polygon', async () => {
                    await polygonTool.selectPolygonTool();
                    await polygonTool.drawPolygon(secondPolygon);
                });

                await test.step('Previously selected annotation is deselected and new one is in edit mode', async () => {
                    await expect(page.locator('[id^="edit-polygon-points-"]')).toHaveCount(1);
                    await expect(annotatorPage.getAnnotationsList().getByLabel('selected annotation')).toHaveCount(1);
                    expect(await annotatorPage.getAnnotationsListItems('annotation polygon')).toHaveLength(2);
                });
            });
        });
    });

    test('Assigns subset to media', async ({ annotatorPage, boundingBoxTool, network }) => {
        const mediaItems = [
            getMockedMediaImage({ id: 'media-1', name: 'item-1.jpg', width: 1920, height: 1080 }),
            getMockedMediaImage({ id: 'media-2', name: 'item-2.jpg', width: 1920, height: 1080 }),
        ];
        let subsetPayload: DatasetSubset | null = 'unassigned';

        const annotationsResponsePerMedia: Record<
            string,
            { annotations: AnnotationDTO[]; user_reviewed: boolean; subset: DatasetSubset }
        > = {
            [mediaItems[0].id]: {
                annotations: [],
                user_reviewed: false,
                subset: 'unassigned',
            },
            [mediaItems[1].id]: {
                annotations: [],
                user_reviewed: false,
                subset: 'validation',
            },
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ params }) => {
                return HttpResponse.json(annotationsResponsePerMedia[params.media_id]);
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: mediaItems,
                    pagination: {
                        offset: 0,
                        limit: 10,
                        count: mediaItems.length,
                        total: mediaItems.length,
                    },
                });
            }),
            http.post(
                '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                async ({ request, params }) => {
                    const payload = await request.json();
                    subsetPayload = payload.subset ?? null;
                    annotationsResponsePerMedia[params.media_id].subset = payload.subset ?? 'unassigned';
                    annotationsResponsePerMedia[params.media_id].annotations = payload.annotations;

                    return HttpResponse.json({});
                }
            )
        );

        await annotatorPage.goto(mockedDetectionProject.id, mediaItems[0].id);

        await test.step('Draw an annotation', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 100, y: 100, width: 150, height: 150 });
        });

        await test.step('Select subset', async () => {
            await annotatorPage.selectSubset('training');
        });

        await test.step('Submit annotations and subset', async () => {
            await annotatorPage.submit();

            expect(subsetPayload).toBe('training');
        });

        await test.step('Navigate to the next media item by clicking in sidebar', async () => {
            await annotatorPage.selectMediaItem(mediaItems[1].name);

            await expect(annotatorPage.getSelectedSubset()).toHaveText('Validation');
        });

        await test.step('Navigate to the previous media item by clicking in sidebar', async () => {
            await annotatorPage.selectMediaItem(mediaItems[0].name);

            await expect(annotatorPage.getSelectedSubset()).toHaveText('Training');
        });
    });

    test.describe('Prediction mode model', () => {
        const olderModel = getMockedModel({
            id: 'older-model-id',
            name: 'Older_Model (older)',
            variants: [
                getMockedVariant({
                    id: 'older-variant-id',
                    format: 'openvino',
                    precision: 'fp32',
                    optimal_confidence_threshold: 0.2,
                }),
            ],
            training_info: {
                status: 'successful',
                label_schema_revision: { labels: [{ id: 'label-1', name: 'cat' }] },
                start_time: '2025-01-01T10:00:00.000000+00:00',
                end_time: '2025-01-01T12:00:00.000000+00:00',
                dataset_revision_id: 'dataset-1',
            },
        });

        const newerModel = getMockedModel({
            id: 'newer-model-id',
            name: 'Newer_Model (newer)',
            variants: [
                getMockedVariant({
                    id: 'newer-variant-id',
                    format: 'openvino',
                    precision: 'fp16',
                    optimal_confidence_threshold: 0.65,
                }),
            ],
            training_info: {
                status: 'successful',
                label_schema_revision: { labels: [{ id: 'label-1', name: 'cat' }] },
                start_time: '2025-02-01T10:00:00.000000+00:00',
                end_time: '2025-02-01T12:00:00.000000+00:00',
                dataset_revision_id: 'dataset-2',
            },
        });

        const emptyPredictHandler = http.post('/api/projects/{project_id}/dataset/media:predict', async () => {
            return HttpResponse.json({ predictions: [{ media: { id: 'item-1' }, prediction: [] }] });
        });

        test('shows no model selector when no models are available', async ({ page, annotatorPage, network }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([]);
                }),
                emptyPredictHandler
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode', async () => {
                await annotatorPage.openPredictionMode();
            });

            await test.step('prediction settings are not available when no models available', async () => {
                await expect(annotatorPage.getPredictionSettingsButton()).toBeHidden();
                await expect(page.getByRole('button', { name: 'Select prediction model' })).toBeHidden();
            });
        });

        test('shows no model selector when models have no OpenVINO variants', async ({
            page,
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([
                        getMockedModel({
                            id: 'pytorch-only-model',
                            variants: [getMockedVariant({ id: 'pytorch-variant', format: 'pytorch' })],
                        }),
                    ]);
                }),
                emptyPredictHandler
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode', async () => {
                await annotatorPage.openPredictionMode();
            });

            await test.step('prediction settings are not available when no OpenVINO models available', async () => {
                await expect(annotatorPage.getPredictionSettingsButton()).toBeHidden();
                await expect(page.getByRole('button', { name: 'Select prediction model' })).toBeHidden();
            });
        });

        test('selects latest model by training end time when no active model is set', async ({
            page,
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([olderModel, newerModel]);
                }),
                emptyPredictHandler
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.openPredictionSettings();
            });

            await test.step('the newer model is pre-selected in the picker', async () => {
                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Newer_Model'
                );
            });
        });

        test('active model takes priority over default latest model selection', async ({
            page,
            annotatorPage,
            network,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([olderModel, newerModel]);
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json({
                        project_id: mockedDetectionProject.id,
                        status: 'idle',
                        source: null,
                        sink: null,
                        // @ts-expect-error We care only about mocking the active model resolution behavior
                        model: olderModel,
                        // @ts-expect-error model_revision_id is not included in getMockedVariant
                        model_variant: getMockedVariant({ id: olderModel.variants[0].id }),
                        device: 'cpu',
                    });
                }),
                emptyPredictHandler
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.openPredictionSettings();
            });

            await test.step('the active model is pre-selected instead of the latest model', async () => {
                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Older_Model'
                );
            });
        });

        test('changing model selection uses the newly selected model for predictions', async ({
            page,
            annotatorPage,
            network,
        }) => {
            let capturedModelVariantId: string | undefined;
            let capturedConfidenceThreshold: number | undefined;

            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([olderModel, newerModel]);
                }),
                http.post('/api/projects/{project_id}/dataset/media:predict', async ({ request }) => {
                    const body = (await request.json()) as unknown as {
                        model_variant_id?: string;
                        confidence_threshold?: number;
                    };
                    capturedModelVariantId = body.model_variant_id;
                    capturedConfidenceThreshold = body.confidence_threshold;

                    return HttpResponse.json({ predictions: [{ media: { id: 'item-1' }, prediction: [] }] });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode — newer model should be selected by default', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.openPredictionSettings();

                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Newer_Model'
                );
            });

            await test.step('select the older model from the picker', async () => {
                const predictResponsePromise = page.waitForResponse((res) => res.url().includes('media:predict'));

                await page.getByRole('button', { name: 'Select prediction model' }).click();
                await page.getByRole('option', { name: /Older_Model/ }).click();

                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Older_Model'
                );

                await predictResponsePromise;
            });

            await test.step('predictions are requested with the newly selected model variant', async () => {
                expect(capturedModelVariantId).toBe(olderModel.variants[0].id);
            });

            // TODO: drop the guard once CONFIDENCE_THRESHOLD is enabled by default
            if (FEATURE_FLAGS.CONFIDENCE_THRESHOLD) {
                await test.step('the confidence threshold follows the selected model', async () => {
                    await expect(page.getByRole('textbox', { name: 'Change Confidence threshold' })).toHaveValue('0.2');
                    expect(capturedConfidenceThreshold).toBe(0.2);
                });
            }
        });

        test('changing device selection uses the new device for predictions', async ({
            page,
            annotatorPage,
            network,
        }) => {
            let capturedDevice: string | undefined;

            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([newerModel]);
                }),
                http.get('/api/system/devices/inference', async () => {
                    return HttpResponse.json([
                        { type: 'cpu', name: 'CPU' },
                        { type: 'xpu', name: 'XPU' },
                    ]);
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json({
                        project_id: mockedDetectionProject.id,
                        status: 'idle',
                        source: null,
                        sink: null,
                        device: 'cpu',
                    });
                }),
                http.post('/api/projects/{project_id}/dataset/media:predict', async ({ request }) => {
                    const body = await request.json();
                    capturedDevice = body.device;

                    return HttpResponse.json({ predictions: [{ media: { id: 'item-1' }, prediction: [] }] });
                })
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode and wait for initial predict request', async () => {
                const predictResponsePromise = page.waitForResponse((res) => res.url().includes('media:predict'));

                await annotatorPage.openPredictionMode();

                await predictResponsePromise;
            });

            await test.step('initial predict request used cpu device', async () => {
                expect(capturedDevice).toBe('cpu');
            });

            await test.step('change device selection to XPU', async () => {
                const predictResponsePromise = page.waitForResponse((res) => res.url().includes('media:predict'));

                await annotatorPage.openPredictionSettings();
                await page.getByRole('button', { name: /inference device/i }).click();
                await page.getByRole('option', { name: /XPU/i }).click();

                await expect(page.getByRole('button', { name: /XPU/i })).toBeVisible();

                await predictResponsePromise;
            });

            await test.step('new predict request used xpu device', async () => {
                expect(capturedDevice).toBe('xpu');
            });
        });

        test('last used model is used by default', async ({ page, annotatorPage, network }) => {
            network.use(
                http.get('/api/projects/{project_id}/models', async () => {
                    return HttpResponse.json([olderModel, newerModel]);
                }),
                emptyPredictHandler
            );

            await annotatorPage.goto(mockedDetectionProject.id, 'item-1');

            await test.step('open prediction mode — newer model is auto-selected by default', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.openPredictionSettings();

                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Newer_Model'
                );
            });

            await test.step('change selection to the older model', async () => {
                const predictResponsePromise = page.waitForResponse((res) => res.url().includes('media:predict'));

                await page.getByRole('button', { name: 'Select prediction model' }).click();
                await page.getByRole('option', { name: /Older_Model/ }).click();

                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Older_Model'
                );

                await predictResponsePromise;
            });

            await test.step('reload the page', async () => {
                await page.reload();
            });

            await test.step('open prediction mode after reload — older model is still selected', async () => {
                await annotatorPage.openPredictionMode();
                await annotatorPage.openPredictionSettings();

                await expect(page.getByRole('button', { name: 'Select prediction model' })).toContainText(
                    'Older_Model'
                );
            });
        });
    });
});
