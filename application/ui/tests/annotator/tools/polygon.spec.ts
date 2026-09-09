// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect } from '@playwright/test';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { Polygon } from '../../../src/shared/types';
import { http, test } from '../../fixtures';
import { withRelative } from '../../utils/mouse';
import { blueLabel, candyBinaryHandler, redLabel } from '../annotator-fixtures';

const mockedDetectionProject = getMockedProject({
    id: '123e4567-e89b-12d3-a456-426614174000',
    task: {
        exclusive_labels: true,
        task_type: 'instance_segmentation',
        labels: [redLabel, blueLabel],
    },
});

const polygonShape: Polygon = {
    type: 'polygon',
    points: [
        { x: 100, y: 100 },
        { x: 250, y: 100 },
        { x: 250, y: 250 },
        { x: 180, y: 300 },
        { x: 100, y: 250 },
    ],
};

test.describe('Polygon', () => {
    test.beforeEach(async ({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedDetectionProject);
            }),
            http.get('/api/projects', () => {
                return HttpResponse.json([mockedDetectionProject]);
            }),
            candyBinaryHandler
        );
    });

    test('Remove points', async ({ page, polygonTool }) => {
        await page.goto(`/projects/${mockedDetectionProject.id}/dataset/item-1`);

        await test.step('Draw an annotation', async () => {
            await polygonTool.selectPolygonTool();
            await polygonTool.drawPolygon(polygonShape);
        });

        await test.step('Remove point with context menu', async ({}) => {
            const relative = await withRelative(page);
            const currentPoint = polygonShape.points[3];
            const relativePoint = relative(currentPoint.x, currentPoint.y);

            await page.getByRole('button', { name: 'Selection' }).click();

            await expect(
                page.getByLabel(`Resize polygon (${currentPoint.x}, ${currentPoint.y}) anchor`)
            ).toBeInViewport();
            await polygonTool.deletePointByRightClick(relativePoint);

            await expect(
                page.getByLabel(`Resize polygon (${currentPoint.x}, ${currentPoint.y}) anchor`)
            ).not.toBeInViewport();
            await expect(page.getByRole('button', { name: 'delete point' })).not.toBeInViewport();
        });
    });

    test('undo/redo behaviour', async ({ page, polygonTool }) => {
        const partialShape: Polygon = {
            type: 'polygon',
            points: [
                { x: 100, y: 100 },
                { x: 250, y: 100 },
                { x: 250, y: 250 },
            ],
        };

        const pointsToString = (points: Polygon['points']): string => {
            return points.map((point) => `${point.x},${point.y}`).join(' ');
        };

        await test.step('Navigate to annotator', async () => {
            await page.goto(`/projects/${mockedDetectionProject.id}/dataset/item-1`);
        });

        await test.step('Draw 3 points without finishing', async () => {
            await polygonTool.selectPolygonTool();
            await polygonTool.drawPolygon(partialShape, { asLasso: false, finishShape: false });
        });

        await test.step('Assert all drawn points are present', async () => {
            await expect(page.getByLabel('new polygon')).toHaveAttribute('points', pointsToString(partialShape.points));
        });

        await test.step('Undo once — last point disappears, previous point remains', async () => {
            await page.getByRole('button', { name: /^undo\b/i }).click();

            const newPolygon = page.getByLabel('new polygon');

            const remainingPoints = partialShape.points.slice(0, -1);
            const lastPoint = partialShape.points[partialShape.points.length - 1];

            await expect(newPolygon).toHaveAttribute('points', pointsToString(remainingPoints));
            await expect(newPolygon).not.toHaveAttribute('points', pointsToString([lastPoint]));
        });

        await test.step('Undo to empty — in-progress polygon is gone', async () => {
            const undoButton = page.getByRole('button', { name: /^undo\b/i });

            await undoButton.click();
            await undoButton.click();

            await expect(page.getByLabel('new polygon')).toBeHidden();
        });

        await test.step('Redo once — a point is restored', async () => {
            await page.getByRole('button', { name: /^redo\b/i }).click();

            const firstPoint = partialShape.points[0];
            await expect(page.getByLabel('new polygon')).toHaveAttribute('points', pointsToString([firstPoint]));
        });

        await test.step('Switch to selection tool — drawing state is reset', async () => {
            await page.getByRole('button', { name: 'Selection' }).click();
            await expect(page.getByLabel('new polygon')).toBeHidden();
        });
    });
});
