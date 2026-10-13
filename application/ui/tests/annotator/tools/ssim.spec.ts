// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect } from '@playwright/test';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { FEATURE_FLAGS } from '../../../src/constants/feature-flags';
import { http, test } from '../../fixtures';
import { candyBinaryHandler, redLabel } from '../annotator-fixtures';

const SCENARIOS = [
    { taskType: 'detection', annotationLabel: 'annotation rect' },
    { taskType: 'instance_segmentation', annotationLabel: 'annotation polygon' },
] as const;

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

        test('Draw a template region and adds annotations for it and its matches', async ({
            page,
            ssimTool,
            annotatorPage,
        }) => {
            await page.goto(`/projects/${mockedProject.id}/dataset`);
            await page.getByRole('img', { name: 'item-1.jpg' }).dblclick();

            await test.step('Select SSIM tool', async () => {
                await ssimTool.selectTool();
            });

            await test.step('Draw a template region', async () => {
                await ssimTool.drawTemplate({ x: 100, y: 100, width: 150, height: 150 });
            });

            await test.step('Expect the template and its matches as annotations', async () => {
                await expect(async () => {
                    const items = await annotatorPage.getAnnotationsListItems(annotationLabel);

                    expect(items.length).toBeGreaterThanOrEqual(2);
                }).toPass({ timeout: 15000 });
            });
        });
    });
}
