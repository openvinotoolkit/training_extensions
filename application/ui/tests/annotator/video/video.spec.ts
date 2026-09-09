// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { AnnotationDTO, MediaListPredictionRequest } from '@/api/types';
import { expect } from '@playwright/test';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedVideoFrame } from 'mocks/mock-media';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http, test } from '../../fixtures';
import { candyPngBuffer, redLabel } from '../annotator-fixtures';
import { ANNOTATIONS_MOCKS, PREDICTIONS_MOCKS } from './mocks';

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'detection',
        labels: [redLabel],
    },
});

const mockAnnotation: AnnotationDTO = {
    shape: {
        type: 'rectangle',
        x: 100,
        y: 100,
        width: 220,
        height: 180,
    },
    labels: [{ id: redLabel.id }],
};

const mockVideoFrame = getMockedVideoFrame({
    id: 'video-1',
    name: 'video-1.mp4',
    frame_number: 0,
    frame_count: 3600,
    frame_stride: 60,
    fps: 60,
    duration: 1000 * 60 * 60,
    width: 960,
    height: 540,
});

const totalFrames = mockVideoFrame.frame_count - 1;

const videoGalleryItem = {
    ...mockVideoFrame,
    video_id: 'video-parent-1',
    frame_index: mockVideoFrame.frame_number,
};

type SubmittedFrameRequest = {
    frameIndex: number;
    annotations: AnnotationDTO[];
};

const dirname = path.dirname(fileURLToPath(import.meta.url));
const videoFilePath = path.resolve(dirname, '../../assets/fish_60.mp4');

test.describe('Annotator video player', () => {
    let frameAnnotations: Record<number, AnnotationDTO[]>;
    let submittedFrameRequests: SubmittedFrameRequest[];

    test.beforeEach(async ({ network }) => {
        frameAnnotations = {
            0: [mockAnnotation],
            1: [],
            2: [],
            3: [],
            4: [],
        };
        submittedFrameRequests = [];

        const videoFile = fs.readFileSync(videoFilePath);

        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            http.get('/api/projects', () => {
                return HttpResponse.json([mockedDetectionProject]);
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: [videoGalleryItem],
                    pagination: { offset: 0, limit: 20, count: 1, total: 1 },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', ({ query }) => {
                if (query.has('frame_index')) {
                    return HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
                        headers: {
                            'Content-Type': 'image/png',
                        },
                    });
                }

                return HttpResponse.arrayBuffer(
                    videoFile.buffer.slice(videoFile.byteOffset, videoFile.byteOffset + videoFile.byteLength),
                    {
                        headers: { 'Content-Type': 'video/mp4', 'Content-Length': videoFile.byteLength.toString() },
                    }
                );
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ request }) => {
                const frameIndex = Number(new URL(request.url).searchParams.get('frame_index') ?? '0');

                return HttpResponse.json({
                    annotations: frameAnnotations[frameIndex] ?? [],
                    user_reviewed: true,
                    subset: 'training',
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/frames', () => {
                return HttpResponse.json(
                    Object.entries(frameAnnotations).map(([frameIndex, annotations]) => ({
                        media_id: mockVideoFrame.id,
                        frame_index: Number(frameIndex),
                        annotation_data: {
                            annotations,
                            subset: 'training' as const,
                            user_reviewed: true,
                        },
                    }))
                );
            }),
            http.post('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ request }) => {
                const url = new URL(request.url);
                const frameIndex = Number(url.searchParams.get('frame_index') ?? '0');
                const body = (await request.json()) as { annotations: AnnotationDTO[] };

                frameAnnotations[frameIndex] = body.annotations;
                submittedFrameRequests.push({ frameIndex, annotations: body.annotations });

                return HttpResponse.json(
                    { annotations: body.annotations, user_reviewed: true, subset: 'training' },
                    { status: 201 }
                );
            })
        );
    });

    test('loads video controls and timeline for a video item', async ({ videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await expect(videoPage.getPlayButton()).toBeVisible();
        await expect(videoPage.getPreviousFrameButton()).toBeDisabled();
        await expect(videoPage.getNextFrameButton()).toBeEnabled();
        await expect(videoPage.getVideoTimeline()).toBeVisible();
        await expect(videoPage.getVideoDuration()).toBeVisible();
    });

    test('toggles play and pause for video playback', async ({ videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await videoPage.play();
        await expect(videoPage.getPauseButton()).toBeVisible();

        await videoPage.pauseVideo();
        await expect(videoPage.getPlayButton()).toBeVisible();
    });

    test('navigates frames and updates frame annotations', async ({ annotatorPage, videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await videoPage.expandToolbar();
        await videoPage.expectCurrentFrame(0, totalFrames);
        expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);

        await videoPage.nextFrame();
        await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride, totalFrames);
        expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);

        await videoPage.previousFrame();
        await videoPage.expectCurrentFrame(0, totalFrames);
        expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
    });

    test('adds annotation on video frame and submits', async ({ annotatorPage, boundingBoxTool, videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);

        await test.step('Selects frame to annotate', async () => {
            await videoPage.expandToolbar();
            await videoPage.nextFrame();
            await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride, totalFrames);
            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(0);
        });

        await test.step('Draws an annotation on video frame', async () => {
            await boundingBoxTool.selectTool();
            await boundingBoxTool.drawBoundingBox({ x: 260, y: 140, width: 180, height: 120 });

            expect(await annotatorPage.getAnnotationsListItems('annotation rect')).toHaveLength(1);
            await expect(videoPage.getSubmitButton()).toBeEnabled();
        });

        await test.step('Submits annotation on video frame', async () => {
            await videoPage.getSubmitButton().click();

            expect(submittedFrameRequests).toHaveLength(1);
            expect(submittedFrameRequests[0].frameIndex).toBe(mockVideoFrame.frame_stride);
            expect(submittedFrameRequests[0].annotations).toHaveLength(1);
            expect(submittedFrameRequests[0].annotations[0].shape.type).toBe('rectangle');
        });

        await test.step('Navigates to the next frame automatically', async () => {
            await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride * 2, totalFrames);
        });
    });

    test('disables next frame button at last frame boundary, disable previous frame at first frame boundary', async ({
        videoPage,
    }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);
        await videoPage.expandToolbar();

        await expect(videoPage.getPreviousFrameButton()).toBeDisabled();

        const lastFrame = mockVideoFrame.frame_count - mockVideoFrame.frame_stride;
        await videoPage.selectFrame(lastFrame);

        await videoPage.expectCurrentFrame(lastFrame, totalFrames);
        await expect(videoPage.getNextFrameButton()).toBeDisabled();
        await expect(videoPage.getPreviousFrameButton()).toBeEnabled();
    });

    test('toggles frame mode and disables frame mode while playing', async ({ videoPage }) => {
        await videoPage.openVideoFromDataset(mockedDetectionProject.id, mockVideoFrame.name);
        await videoPage.expandToolbar();

        // In 1/1 mode, frame step follows default frame_stride/fps behavior (here, +2).
        await expect(videoPage.getFrameModeIndicator()).toHaveText('1/1');
        await videoPage.nextFrame();
        await videoPage.expectCurrentFrame(mockVideoFrame.frame_stride, totalFrames);
        await videoPage.previousFrame();
        await videoPage.expectCurrentFrame(0, totalFrames);

        // In ALL mode, frame step is 1 frame.
        await videoPage.toggleFrameMode();
        await expect(videoPage.getFrameModeIndicator()).toHaveText('ALL');
        await videoPage.nextFrame();
        await videoPage.expectCurrentFrame(1, totalFrames);

        await videoPage.play();
        await expect(videoPage.getPauseButton()).toBeVisible();
        await expect(videoPage.getToggleFrameModeButton()).toBeDisabled();

        await videoPage.pauseVideo();
        await expect(videoPage.getPlayButton()).toBeVisible();
        await expect(videoPage.getToggleFrameModeButton()).toBeEnabled();

        await videoPage.toggleFrameMode();
        await expect(videoPage.getFrameModeIndicator()).toHaveText('1/1');
    });

    test('Displays annotations and predictions in the segments below video timeline', async ({
        videoPage,
        annotatorPage,
        network,
    }) => {
        const fishLabel = getMockedLabel({
            id: 'a6efefed-e469-4b1c-b803-c2e21ea0597b',
            name: 'Fish',
            color: '#ad2323',
        });

        const mockedProject = getMockedProject({
            id: '123e4567-e89b-12d3-a456-426614174000',
            task: {
                exclusive_labels: true,
                task_type: 'instance_segmentation',
                labels: [fishLabel],
            },
        });

        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedProject);
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/frames', async () => {
                return HttpResponse.json(ANNOTATIONS_MOCKS);
            }),
            http.post('/api/projects/{project_id}/dataset/media:predict', async () => {
                return HttpResponse.json({
                    predictions: PREDICTIONS_MOCKS,
                });
            })
        );

        await videoPage.openVideoFromDataset(mockedProject.id, mockVideoFrame.name);
        await videoPage.expandToolbar();

        await expect(annotatorPage.getAnnotatorMode('annotation')).toHaveAttribute('aria-pressed', 'true');

        await Promise.all(
            ANNOTATIONS_MOCKS.map(async (annotation) => {
                await expect(videoPage.getLabelSegment(annotation.frame_index, fishLabel.name)).toBeVisible();
            })
        );

        await annotatorPage.openPredictionMode();

        await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');

        await Promise.all(
            PREDICTIONS_MOCKS.map(async ({ media }) => {
                await expect(videoPage.getLabelSegment(Number(media.frame_index), fishLabel.name)).toBeVisible();
            })
        );
    });

    test('Prefetches predictions for the next video frame', async ({ videoPage, annotatorPage, network }) => {
        const fishLabel = getMockedLabel({ id: 'a6efefed-e469-4b1c-b803-c2e21ea0597b', name: 'Fish' });

        const mockedProject = getMockedProject({
            task: {
                exclusive_labels: true,
                task_type: 'instance_segmentation',
                labels: [fishLabel],
            },
        });

        const predictRequests: MediaListPredictionRequest[] = [];

        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedProject);
            }),
            http.get('/api/projects', () => {
                return HttpResponse.json([mockedProject]);
            }),
            http.get('/api/projects/{project_id}/models', async () => {
                return HttpResponse.json([getMockedModel({ variants: [getMockedVariant({})] })]);
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/frames', async () => {
                return HttpResponse.json(ANNOTATIONS_MOCKS);
            }),
            http.post('/api/projects/{project_id}/dataset/media:predict', async ({ request }) => {
                const body = (await request.json()) as MediaListPredictionRequest;
                predictRequests.push(body);

                return HttpResponse.json({
                    predictions: PREDICTIONS_MOCKS,
                });
            })
        );

        const currentFrame = mockVideoFrame.frame_number;
        const nextFrame = mockVideoFrame.frame_number + mockVideoFrame.frame_stride;

        const isRequestForFrame = (body: MediaListPredictionRequest, frameNumber: number) =>
            body.media.some(
                ({ media_id, range }) =>
                    media_id === mockVideoFrame.id &&
                    range?.stride === mockVideoFrame.frame_stride &&
                    range?.end_frame === frameNumber &&
                    range?.start_frame === frameNumber
            );

        await test.step('Opens the video in prediction mode', async () => {
            await videoPage.openVideoFromDataset(mockedProject.id, mockVideoFrame.name);
            await videoPage.expandToolbar();
            await annotatorPage.openPredictionMode();
            await expect(annotatorPage.getAnnotatorMode('prediction')).toHaveAttribute('aria-pressed', 'true');
        });

        await test.step('Fetches predictions for the current frame and prefetches the next frame', async () => {
            await expect.poll(() => predictRequests.some((body) => isRequestForFrame(body, currentFrame))).toBe(true);
            await expect.poll(() => predictRequests.some((body) => isRequestForFrame(body, nextFrame))).toBe(true);
        });

        await test.step('Reuses the prefetched cache entry when navigating to the next frame', async () => {
            const frameAfterNext = nextFrame + mockVideoFrame.frame_stride;
            const requestsBefore = predictRequests.filter((body) => isRequestForFrame(body, nextFrame)).length;

            await videoPage.nextFrame();
            await videoPage.expectCurrentFrame(nextFrame, totalFrames);

            // Wait for a deterministic signal that the navigation has been fully handled:
            // once we are on `nextFrame`, the player should prefetch the following frame.
            await expect.poll(() => predictRequests.some((body) => isRequestForFrame(body, frameAfterNext))).toBe(true);

            const requestsAfter = predictRequests.filter((body) => isRequestForFrame(body, nextFrame)).length;
            expect(requestsAfter).toBe(requestsBefore);
        });
    });
});
