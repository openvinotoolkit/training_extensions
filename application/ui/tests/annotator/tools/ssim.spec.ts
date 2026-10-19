// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect } from '@playwright/test';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { FEATURE_FLAGS } from '../../../src/constants/feature-flags';
import type { AnnotatorPage } from '../../datasets/annotator-page';
import { http, test } from '../../fixtures';
import { candyBinaryHandler, redLabel } from '../annotator-fixtures';

const SCENARIOS = [
    { taskType: 'detection', annotationLabel: 'annotation rect' },
    { taskType: 'instance_segmentation', annotationLabel: 'annotation polygon' },
] as const;

const expectAnnotationsCount = async (
    annotatorPage: AnnotatorPage,
    annotationLabel: (typeof SCENARIOS)[number]['annotationLabel'],
    minimum: number
) => {
    await expect(async () => {
        const items = await annotatorPage.getAnnotationsListItems(annotationLabel);

        expect(items.length).toBeGreaterThanOrEqual(minimum);
    }).toPass({ timeout: 5000 });
};

for (const { taskType, annotationLabel } of SCENARIOS) {
    const mockedProject = getMockedProject({
        id: '123e4567-e89b-12d3-a456-426614174002',
        task: {
            exclusive_labels: true,
            task_type: taskType,
            labels: [redLabel],
        },
    });

    test.describe(`SSIM tool (${taskType})`, () => {
        test.skip(!FEATURE_FLAGS.SSIM_TOOL, 'SSIM tool is behind the SSIM_TOOL feature flag');

        test.beforeEach(async ({ network }) => {
            network.use(
                http.get('/api/projects/{project_id}', () => {
                    return HttpResponse.json(mockedProject);
                }),
                candyBinaryHandler
            );
        });

        test('Previews the template and its matches and adds them on accept', async ({
            page,
            ssimTool,
            annotatorPage,
        }) => {
            await page.goto(`/projects/${mockedProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('Select SSIM tool', async () => {
                await ssimTool.selectTool();
                await expect(ssimTool.getAcceptButton()).toBeDisabled();
                await expect(ssimTool.getRejectButton()).toBeDisabled();
            });

            await test.step('Draw a template region', async () => {
                await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
                await ssimTool.waitForPreview();
            });

            await test.step('Expect a preview without annotations', async () => {
                expect(await ssimTool.getPreviewShapes().count()).toBeGreaterThanOrEqual(2);
                expect(await annotatorPage.getAnnotationsListItems(annotationLabel)).toHaveLength(0);
            });

            await test.step('Accept the preview', async () => {
                await ssimTool.getAcceptButton().click();

                await expectAnnotationsCount(annotatorPage, annotationLabel, 2);
                await expect(ssimTool.getPreviewShapes()).toHaveCount(0);
            });
        });

        test('Limits the matches with the number of items slider', async ({ page, ssimTool, annotatorPage }) => {
            await page.goto(`/projects/${mockedProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await ssimTool.selectTool();
            await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
            await ssimTool.waitForPreview();

            await ssimTool.getNumberOfItemsSlider().focus();
            await page.keyboard.press('Home');

            await expect(ssimTool.getNumberOfItemsSlider()).toHaveValue('1');
            await expect(ssimTool.getPreviewShapes()).toHaveCount(1);

            await page.keyboard.press('Enter');

            await expect(async () => {
                expect(await annotatorPage.getAnnotationsListItems(annotationLabel)).toHaveLength(1);
            }).toPass({ timeout: 5000 });
        });

        test('Discards the preview on reject', async ({ page, ssimTool, annotatorPage }) => {
            await page.goto(`/projects/${mockedProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await ssimTool.selectTool();
            await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
            await ssimTool.waitForPreview();

            await ssimTool.getRejectButton().click();

            await expect(ssimTool.getPreviewShapes()).toHaveCount(0);
            expect(await annotatorPage.getAnnotationsListItems(annotationLabel)).toHaveLength(0);
        });

        test('Accepts with Enter and rejects with Escape without leaving the annotator', async ({
            page,
            ssimTool,
            annotatorPage,
        }) => {
            await page.goto(`/projects/${mockedProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();
            await ssimTool.selectTool();

            await test.step('Escape rejects the preview', async () => {
                await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
                await ssimTool.waitForPreview();

                await page.keyboard.press('Escape');

                await expect(ssimTool.getPreviewShapes()).toHaveCount(0);
                await expect(ssimTool.getOptions()).toBeVisible();
                expect(await annotatorPage.getAnnotationsListItems(annotationLabel)).toHaveLength(0);
            });

            await test.step('Enter accepts the preview', async () => {
                await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
                await ssimTool.waitForPreview();

                await page.keyboard.press('Enter');

                await expectAnnotationsCount(annotatorPage, annotationLabel, 2);
                await expect(ssimTool.getPreviewShapes()).toHaveCount(0);
                await expect(ssimTool.getOptions()).toBeVisible();
            });
        });
    });
}
