// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import type { AnnotationDTO, Project } from '@/api/types';
import { NetworkFixture } from '@msw/playwright';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedMediaImage, getMockedVideo, getMultipleMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { v4 as uuid } from 'uuid';

import { expect, http, test } from '../fixtures';

const mockedItems = getMultipleMockedMediaImage(40, '1');
const mockedItems2 = getMultipleMockedMediaImage(20, '2');
const totalElements = mockedItems.length + mockedItems2.length;

const dirname = path.dirname(fileURLToPath(import.meta.url));
const sampleImagePath = path.resolve(dirname, '../assets/candy-thumbnail.png');
const sampleImageBuffer = fs.readFileSync(sampleImagePath);

const sampleVideoPath = path.resolve(dirname, '../assets/fish_60.mp4');

test.describe('Dataset', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', ({}) => {
                return HttpResponse.arrayBuffer(sampleImageBuffer.buffer, {
                    headers: { 'Content-Type': 'image/png' },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media', ({ query }) => {
                const offset = Number(query.get('offset') ?? 0);
                const limit = Number(query.get('limit'));
                const items = offset === 0 ? mockedItems : mockedItems2;

                return HttpResponse.json({
                    items,
                    pagination: {
                        offset,
                        limit,
                        count: items.length,
                        total: totalElements,
                    },
                });
            })
        );
    });

    test('list items', async ({ page, datasetPage }) => {
        const waitForBatch = (offset: number) =>
            page.waitForResponse(
                (response) => response.url().includes('/dataset/media') && response.url().includes(`offset=${offset}`)
            );

        const batch40 = waitForBatch(40);

        await datasetPage.goto();

        await expect(datasetPage.getImagesCountText(totalElements)).toBeVisible();

        await datasetPage.getMediaGrid().press('End');
        await batch40;

        await datasetPage.selectAll();

        await expect(datasetPage.getSelectedCountText(totalElements)).toBeVisible();
    });

    test('select multiple images', async ({ datasetPage }) => {
        const selectedElements = 5;

        await datasetPage.goto();

        await expect(datasetPage.getImagesCountText(totalElements)).toBeVisible();

        for (let i = 0; i < selectedElements; i++) {
            await datasetPage.selectMediaItem(mockedItems[i].id);
        }

        await expect(datasetPage.getSelectedCountText(selectedElements)).toBeVisible();
    });

    test('loads additional items when scrolling to the end of the container', async ({ datasetPage }) => {
        await datasetPage.goto();

        await expect(datasetPage.getImagesCountText(totalElements)).toBeVisible();

        await datasetPage.getMediaGrid().press('End');

        await expect(datasetPage.getImagesCountText(totalElements)).toBeVisible();
    });

    test('selected media item is saved in the URL', async ({ page, annotatorPage, datasetPage }) => {
        const [firstElement] = mockedItems;
        await datasetPage.goto();
        await expect(annotatorPage.getAnnotationsList()).not.toBeInViewport();

        await datasetPage.dblClickMediaItem(firstElement.name);

        await expect(annotatorPage.getAnnotationsList()).toBeInViewport();

        expect(page.url()).toContain(`/dataset/${firstElement.id}`);
    });

    test('upload shows start, progress and finish toasts', async ({ network, datasetPage }) => {
        let uploadRequestCount = 0;
        const firstBatchCount = 10;
        const totalFiles = firstBatchCount + 1;

        network.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                uploadRequestCount += 1;
                const isLastUploadRequest = uploadRequestCount === totalFiles;

                // Small delay to test the "in progress" toast or else we would only see start and finish toasts
                await new Promise((resolve) => setTimeout(resolve, isLastUploadRequest ? 250 : 30));

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), {
                    status: 201,
                });
            })
        );

        await datasetPage.goto();

        await datasetPage.uploadFiles(
            Array.from({ length: totalFiles }, (_, index) => ({
                name: `upload-${index + 1}.png`,
                mimeType: 'image/png',
                buffer: sampleImageBuffer,
            }))
        );

        await expect(datasetPage.getUploadButton()).toBeDisabled();

        await expect(datasetPage.getUploadProgressText(totalFiles)).toBeVisible();

        await expect(datasetPage.getUploadFinishedText(totalFiles)).toBeVisible();
    });

    test('shows per-file upload details dialog with status', async ({ network, datasetPage }) => {
        const filesToUpload = [
            { name: 'image-one.png', mimeType: 'image/png', buffer: sampleImageBuffer },
            { name: 'image-two.png', mimeType: 'image/png', buffer: sampleImageBuffer },
        ];

        network.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        await datasetPage.goto();

        await datasetPage.uploadFiles(filesToUpload);

        // Wait for the final toast so the toast element is stable (in-progress toast re-renders on each update).
        await expect(datasetPage.getUploadFinishedText(2)).toBeVisible();

        await datasetPage.clickShowDetails();

        const dialog = datasetPage.getUploadDetailsDialog();
        await expect(dialog).toBeVisible();

        // One header row + two body rows.
        await expect(datasetPage.getUploadDetailsRows()).toHaveCount(3);

        await expect(dialog.getByRole('row', { name: /image-one\.png/ })).toContainText('Uploaded');
        await expect(dialog.getByRole('row', { name: /image-two\.png/ })).toContainText('Uploaded');

        await datasetPage.closeUploadDetailsDialog();
        await expect(dialog).toBeHidden();
    });

    test.describe('Bulk labelling while uploading media items', () => {
        const mockedImages = [getMockedMediaImage({ id: uuid() }), getMockedMediaImage({ id: uuid() })];
        const mockedVideo = getMockedVideo({ id: uuid() });
        const mockedMedia = [...mockedImages, mockedVideo];
        const mockedLabels = [
            getMockedLabel({
                id: 'id-cat',
                name: 'cat',
            }),
            getMockedLabel({
                id: 'id-dog',
                name: 'dog',
            }),
        ];
        const sampleVideoBuffer = fs.readFileSync(sampleVideoPath);

        const filesToUpload = [
            ...mockedImages.map((_, idx) => ({
                name: `upload-${idx + 1}.png`,
                mimeType: 'image/png',
                buffer: sampleImageBuffer,
            })),
            { name: 'upload-video.mp4', mimeType: 'video/mp4', buffer: sampleVideoBuffer },
        ];

        const mockNetwork = (
            network: NetworkFixture,
            project: Project,
            options?: { onUpload?: () => Promise<void> }
        ) => {
            const createAnnotationPayloads: [string, AnnotationDTO[]][] = [];
            let getMediaCount = 0;

            network.use(
                http.post('/api/projects/{project_id}/dataset/media', async () => {
                    const media = mockedMedia[getMediaCount];

                    getMediaCount++;

                    await options?.onUpload?.();

                    return HttpResponse.json(media, {
                        status: 201,
                    });
                }),
                http.get('/api/projects/{project_id}', async () => {
                    return HttpResponse.json(project);
                }),
                http.post(
                    '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                    async ({ request, params }) => {
                        const payload = await request.json();

                        createAnnotationPayloads.push([params.media_id, payload.annotations]);

                        return HttpResponse.json({
                            annotations: payload.annotations,
                            user_reviewed: true,
                            subset: 'training',
                        });
                    }
                )
            );

            return createAnnotationPayloads;
        };

        test('Single label: bulk labelling is only enabled for classification task and only for images', async ({
            network,
            datasetPage,
        }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: true,
                        labels: mockedLabels,
                    },
                })
            );

            await datasetPage.goto();

            await datasetPage.uploadFiles(filesToUpload);

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.selectLabel(mockedLabels[0].name);

            await datasetPage.clickContinue();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [
                        mockedImages[0].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }],
                            },
                        ],
                    ],
                    [
                        mockedImages[1].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }],
                            },
                        ],
                    ],
                ]);
            }).toPass();
        });

        test('Multi label: bulk labelling is only enabled for classification task and only for images', async ({
            network,
            datasetPage,
        }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: false,
                        labels: mockedLabels,
                    },
                })
            );

            await datasetPage.goto();

            await datasetPage.uploadFiles(filesToUpload);

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.selectLabel(mockedLabels[0].name);
            await datasetPage.selectLabel(mockedLabels[1].name);

            await datasetPage.clickContinue();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [
                        mockedImages[0].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }, { id: mockedLabels[1].id }],
                            },
                        ],
                    ],
                    [
                        mockedImages[1].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }, { id: mockedLabels[1].id }],
                            },
                        ],
                    ],
                ]);
            }).toPass();
        });

        test('Multi label: empty label creates empty annotations', async ({ network, datasetPage }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: false,
                        labels: mockedLabels,
                    },
                })
            );

            await datasetPage.goto();

            await datasetPage.uploadFiles(filesToUpload);

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.selectLabel('No label');

            await datasetPage.clickContinue();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [mockedImages[0].id, []],
                    [mockedImages[1].id, []],
                ]);
            }).toPass();
        });

        test('Skip closes the dialog without waiting for the upload to finish', async ({ network, datasetPage }) => {
            let releaseUpload: () => void = () => {};
            const uploadGate = new Promise<void>((resolve) => {
                releaseUpload = resolve;
            });

            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: true,
                        labels: mockedLabels,
                    },
                }),
                { onUpload: () => uploadGate }
            );

            await datasetPage.goto();

            await datasetPage.uploadFiles(filesToUpload);

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.clickSkip();

            await expect(datasetPage.getLabelAssignmentHeading()).toBeHidden();

            releaseUpload();
            await expect(datasetPage.getUploadFinishedText(filesToUpload.length)).toBeVisible();

            expect(createAnnotationPayloads).toEqual([]);
        });
    });

    test.describe('Bulk labelling for selected images', () => {
        const mockedImages = [
            getMockedMediaImage({ id: uuid(), name: 'media-1' }),
            getMockedMediaImage({ id: uuid(), name: 'media-2' }),
        ];
        const mockedVideo = getMockedVideo({ id: uuid(), name: 'media-3' });
        const mockedMedia = [...mockedImages, mockedVideo];
        const mockedLabels = [
            getMockedLabel({
                id: 'id-cat',
                name: 'cat',
            }),
            getMockedLabel({
                id: 'id-dog',
                name: 'dog',
            }),
        ];

        const mockNetwork = (network: NetworkFixture, project: Project) => {
            const createAnnotationPayloads: [string, AnnotationDTO[]][] = [];

            network.use(
                http.get('/api/projects/{project_id}/dataset/media', () => {
                    return HttpResponse.json({
                        items: mockedMedia,
                        pagination: {
                            offset: 0,
                            limit: 10,
                            count: mockedMedia.length,
                            total: mockedMedia.length,
                        },
                    });
                }),
                http.get('/api/projects/{project_id}', async () => {
                    return HttpResponse.json(project);
                }),
                http.post(
                    '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                    async ({ request, params }) => {
                        const payload = await request.json();

                        createAnnotationPayloads.push([params.media_id, payload.annotations]);

                        return HttpResponse.json({
                            annotations: payload.annotations,
                            user_reviewed: true,
                            subset: 'training',
                        });
                    }
                )
            );

            return createAnnotationPayloads;
        };

        test('Single label: bulk labelling is only enabled for classification task and only for images', async ({
            network,
            datasetPage,
        }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: true,
                        labels: mockedLabels,
                    },
                })
            );

            await datasetPage.goto();

            await expect(datasetPage.getAssignLabelButton()).toBeHidden();

            await datasetPage.selectMediaItem(mockedMedia[0].id);
            await datasetPage.selectMediaItem(mockedMedia[1].id);

            await datasetPage.clickAssignLabel();

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.selectLabel(mockedLabels[0].name);

            await datasetPage.clickDialogAssign();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [
                        mockedImages[0].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }],
                            },
                        ],
                    ],
                    [
                        mockedImages[1].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }],
                            },
                        ],
                    ],
                ]);
            }).toPass();
        });

        test('Multi label: bulk labelling is only enabled for classification task and only for images', async ({
            network,
            datasetPage,
        }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: false,
                        labels: mockedLabels,
                    },
                })
            );

            await datasetPage.goto();

            await datasetPage.selectMediaItem(mockedMedia[0].id);
            await datasetPage.selectMediaItem(mockedMedia[1].id);
            await datasetPage.selectMediaItem(mockedMedia[2].id);

            await datasetPage.clickAssignLabel();

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.selectLabel(mockedLabels[0].name);
            await datasetPage.selectLabel(mockedLabels[1].name);

            await datasetPage.clickDialogAssign();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [
                        mockedImages[0].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }, { id: mockedLabels[1].id }],
                            },
                        ],
                    ],
                    [
                        mockedImages[1].id,
                        [
                            {
                                shape: {
                                    type: 'full_image',
                                },
                                labels: [{ id: mockedLabels[0].id }, { id: mockedLabels[1].id }],
                            },
                        ],
                    ],
                ]);
            }).toPass();
        });

        test('Multi label: empty label creates empty annotations', async ({ network, datasetPage }) => {
            const createAnnotationPayloads = mockNetwork(
                network,
                getMockedProject({
                    task: {
                        task_type: 'classification',
                        exclusive_labels: false,
                        labels: mockedLabels,
                    },
                })
            );

            await datasetPage.goto();

            await datasetPage.selectMediaItem(mockedMedia[0].id);
            await datasetPage.selectMediaItem(mockedMedia[1].id);
            await datasetPage.selectMediaItem(mockedMedia[2].id);

            await datasetPage.clickAssignLabel();

            await expect(datasetPage.getLabelAssignmentHeading()).toBeVisible();

            await datasetPage.selectLabel(mockedLabels[0].name);
            await datasetPage.selectLabel('No label');

            await datasetPage.clickDialogAssign();

            await expect(() => {
                expect(createAnnotationPayloads).toEqual([
                    [mockedImages[0].id, []],
                    [mockedImages[1].id, []],
                ]);
            }).toPass();
        });
    });
});
