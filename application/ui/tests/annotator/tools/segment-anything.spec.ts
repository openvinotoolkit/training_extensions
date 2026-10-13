// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '@/api/types';
import { expect } from '@playwright/test';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http, test } from '../../fixtures';
import { candyBinaryHandler, redLabel } from '../annotator-fixtures';

// Matches the size of `candy.png` and of the default mocked media item.
const IMAGE_SIZE = { width: 1000, height: 750 };
const EMBEDDING_BYTES = 256 * 64 * 64 * Float32Array.BYTES_PER_ELEMENT;

// The real decoder still returns a blob around the prompt point for an all-zero embedding,
// so the test exercises the whole worker + decoder pipeline without a binary fixture.
const buildEmbeddingPayload = ({ width, height }: typeof IMAGE_SIZE): ArrayBuffer => {
    const scale = 1024 / Math.max(width, height);
    const header = new TextEncoder().encode(
        JSON.stringify({
            __metadata__: {
                original_width: String(width),
                original_height: String(height),
                new_width: String(Math.ceil(width * scale)),
                new_height: String(Math.ceil(height * scale)),
            },
            image_embeddings: { dtype: 'F32', shape: [1, 256, 64, 64], data_offsets: [0, EMBEDDING_BYTES] },
        })
    );

    const payload = new ArrayBuffer(8 + header.byteLength + EMBEDDING_BYTES);
    new DataView(payload).setBigUint64(0, BigInt(header.byteLength), true);
    new Uint8Array(payload, 8, header.byteLength).set(header);

    return payload;
};

const isPointInShape = (shape: AnnotationDTO['shape'], { x, y }: { x: number; y: number }) => {
    if (shape.type === 'rectangle') {
        return x >= shape.x && x <= shape.x + shape.width && y >= shape.y && y <= shape.y + shape.height;
    }

    if (shape.type === 'polygon') {
        const points = shape.points;
        let inside = false;

        for (let index = 0, previous = points.length - 1; index < points.length; previous = index++) {
            const currentPoint = points[index];
            const previousPoint = points[previous];
            const intersects =
                currentPoint.y > y !== previousPoint.y > y &&
                x <
                    ((previousPoint.x - currentPoint.x) * (y - currentPoint.y)) / (previousPoint.y - currentPoint.y) +
                        currentPoint.x;

            if (intersects) inside = !inside;
        }

        return inside;
    }

    return false;
};

const getMockedSamProject = (taskType: 'detection' | 'instance_segmentation') =>
    getMockedProject({
        id: `sam-${taskType}`,
        task: { exclusive_labels: true, task_type: taskType, labels: [redLabel] },
    });

const CASES = [
    { taskType: 'detection', shapeType: 'rectangle', annotationLabel: 'annotation rect' },
    { taskType: 'instance_segmentation', shapeType: 'polygon', annotationLabel: 'annotation polygon' },
] as const;

test.describe('Segment anything tool', () => {
    for (const { taskType, shapeType, annotationLabel } of CASES) {
        test(`${taskType}: hovering previews a ${shapeType} and clicking adds it`, async ({
            page,
            network,
            annotatorPage,
        }) => {
            const project = getMockedSamProject(taskType);
            const prompt = { x: 500, y: 375 };

            network.use(
                http.get('/api/projects/{project_id}', () => HttpResponse.json(project)),
                candyBinaryHandler,
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/embeddings', () =>
                    HttpResponse.arrayBuffer(buildEmbeddingPayload(IMAGE_SIZE), {
                        headers: { 'Content-Type': 'application/octet-stream' },
                    })
                ),
                http.post('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ request }) => {
                    const body = (await request.json()) as { annotations: AnnotationDTO[] };

                    return HttpResponse.json(
                        { annotations: body.annotations, user_reviewed: true, subset: 'training' },
                        { status: 201 }
                    );
                })
            );

            await annotatorPage.goto(project.id, 'item-1');

            await test.step('Select the tool and wait for the model', async () => {
                await page.getByRole('button', { name: 'Auto segmentation' }).click();

                await expect(page.getByLabel('SAM tool canvas')).toBeVisible({ timeout: 60_000 });
                await expect(annotatorPage.getProcessingImage()).toBeHidden();
            });

            await test.step('Hover over the image and click to accept the preview', async () => {
                await annotatorPage.annotateAt(prompt.x, prompt.y);

                await expect(async () => {
                    expect(await annotatorPage.getAnnotationsListItems(annotationLabel)).toHaveLength(1);
                }).toPass();
            });

            await test.step('Submit the annotation around the prompt point', async () => {
                const response = await annotatorPage.submitAndWaitForSave();
                const { annotations } = response.request().postDataJSON() as { annotations: AnnotationDTO[] };

                expect(annotations).toHaveLength(1);
                expect(annotations[0].shape.type).toBe(shapeType);
                expect(annotations[0].labels).toEqual([{ id: redLabel.id }]);
                expect(isPointInShape(annotations[0].shape, prompt)).toBe(true);
            });
        });
    }

    test('shows an error when the image embedding cannot be fetched', async ({ page, network, annotatorPage }) => {
        const project = getMockedSamProject('instance_segmentation');

        network.use(
            http.get('/api/projects/{project_id}', () => HttpResponse.json(project)),
            candyBinaryHandler,
            http.get(
                '/api/projects/{project_id}/dataset/media/{media_id}/embeddings',
                () => new HttpResponse(null, { status: 404 })
            )
        );

        await annotatorPage.goto(project.id, 'item-1');
        await page.getByRole('button', { name: 'Auto segmentation' }).click();

        // Client errors are not retried, so the error shows up without the retry back-off.
        await expect(
            page.getByText(/Error in Segment Anything tool: Could not fetch the image embedding \(404/)
        ).toBeVisible({ timeout: 5_000 });
    });
});
