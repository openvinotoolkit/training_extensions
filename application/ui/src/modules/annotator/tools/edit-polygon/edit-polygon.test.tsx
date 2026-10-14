// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import '@wessberg/pointer-events';

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { Annotation, Point, Polygon } from '../../../../shared/types';
import { AnnotationVisibilityProvider } from '../../annotation-visibility-provider.component';
import { CanvasSettingsProvider } from '../../shell/primary-toolbar/settings/canvas-settings-provider.component';
import { removeOffLimitPointsPolygon } from '../utils';
import { EditPolygon } from './edit-polygon.component';

const mockROI = { x: 0, y: 0, width: 1000, height: 1000 };

vi.mock('../../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: vi.fn(() => ({ roi: mockROI })),
}));
vi.mock('../utils', async (importActual) => {
    const actual = await importActual<typeof import('../utils')>();
    return {
        ...actual,
        removeOffLimitPointsPolygon: vi.fn((shape) => shape),
    };
});

const mockedUpdateAnnotations = vi.fn();
const mockedDeleteAnnotations = vi.fn();
vi.mock('../../annotation-document-provider.component', async (importActual) => {
    const actual = await importActual<typeof import('../../annotation-document-provider.component')>();
    return {
        ...actual,
        useAnnotationCommands: vi.fn(() => ({
            updateAnnotations: mockedUpdateAnnotations,
            deleteAnnotations: mockedDeleteAnnotations,
        })),
    };
});

const moveLine = (rect: HTMLElement, startPoint: Point, endPoint: Point) => {
    fireEvent.pointerDown(rect, {
        buttons: 1,
        clientX: startPoint.x,
        clientY: startPoint.y,
    });

    fireEvent.pointerMove(rect, {
        buttons: 1,
        clientX: endPoint.x,
        clientY: endPoint.y,
    });

    fireEvent.pointerUp(rect, {
        buttons: 1,
        clientX: endPoint.x,
        clientY: endPoint.y,
    });
};

const renderApp = async (
    annotation: Annotation & {
        shape: {
            type: 'polygon';
        };
    }
) => {
    const app = render(
        <AnnotationVisibilityProvider>
            <CanvasSettingsProvider>
                <EditPolygon annotation={annotation} zoom={1} />
            </CanvasSettingsProvider>
        </AnnotationVisibilityProvider>
    );

    await waitFor(() => {
        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });

    return app;
};

describe('EditPolygonTool', () => {
    const annotation = getMockedAnnotation({
        id: 'polygon-1',
        shape: {
            type: 'polygon',
            points: [
                { x: 20, y: 10 },
                { x: 70, y: 30 },
                { x: 80, y: 90 },
            ],
        },
    }) as Annotation & { shape: Polygon };
    const shape = annotation.shape;

    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({}));
            })
        );

        vi.clearAllMocks();
    });

    it('allows the user to translate a polygon', async () => {
        await renderApp(annotation);

        // Move the shape
        const startPoint = { x: 90, y: 10 };
        const endPoint = { x: 80, y: 0 };

        const rect = screen.getByLabelText('Drag to move shape');
        moveLine(rect, startPoint, endPoint);

        const translate = {
            x: startPoint.x - endPoint.x,
            y: startPoint.y - endPoint.y,
        };

        const finalShape = {
            ...shape,
            points: shape.points.map((point) => ({
                x: point.x - translate.x,
                y: point.y - translate.y,
            })),
        };

        expect(removeOffLimitPointsPolygon).toHaveBeenCalledWith(finalShape, mockROI);
        expect(mockedUpdateAnnotations).toHaveBeenCalledWith([
            {
                ...annotation,
                shape: finalShape,
            },
        ]);
    });

    it("allows the user to move one of the polygon's anchor points", async () => {
        const startPoint = { x: 70, y: 30 };
        const translate = { x: 10, y: 10 };

        const endPoint = {
            x: startPoint.x + translate.x,
            y: startPoint.y + translate.y,
        };

        await renderApp(annotation);

        const rect = screen.getByLabelText(`Resize polygon (${startPoint.x}, ${startPoint.y}) anchor`);
        moveLine(rect, startPoint, endPoint);

        const finalShape = {
            ...shape,
            points: [
                { x: 20, y: 10 },
                { x: 80, y: 40 },
                { x: 80, y: 90 },
            ],
        };
        expect(removeOffLimitPointsPolygon).toHaveBeenCalledWith(finalShape, mockROI);
        expect(mockedUpdateAnnotations).toHaveBeenCalledWith([
            {
                ...annotation,
                shape: finalShape,
            },
        ]);
    });

    describe('adding a point to an existing polygon', () => {
        const mockAnnotation = getMockedAnnotation({
            id: 'polygon-1',
            shape: {
                type: 'polygon',
                points: [
                    { x: 10, y: 10 },
                    { x: 50, y: 50 },
                    { x: 10, y: 50 },
                ],
            },
        }) as Annotation & { shape: Polygon };
        const mockShape = mockAnnotation.shape;

        const hoverPoint = { clientX: 40, clientY: 50 };

        it('can add a point between two lines', async () => {
            const expectedProjectedPoint = { x: 45, y: 45 };

            await renderApp(mockAnnotation);

            const line = screen.getByLabelText('Line between point 0 and 1');
            fireEvent.pointerMove(line, { ...hoverPoint });

            const ghostPoint = screen.getByLabelText('Add a point between point 0 and 1');
            fireEvent.pointerDown(ghostPoint, { buttons: 1, ...hoverPoint });
            fireEvent.pointerUp(ghostPoint, { buttons: 1, ...hoverPoint });

            const finalShape = {
                ...mockShape,
                points: [mockShape.points[0], expectedProjectedPoint, mockShape.points[1], mockShape.points[2]],
            };
            expect(removeOffLimitPointsPolygon).not.toHaveBeenCalled();

            expect(mockedUpdateAnnotations).toHaveBeenCalledWith([
                {
                    ...mockAnnotation,
                    shape: finalShape,
                },
            ]);
        });

        it('can move the new point before adding it', async () => {
            const expectedProjectedPoint = { x: 45, y: 45 };
            const moveBy = { x: 5, y: 15 };

            await renderApp(mockAnnotation);

            const line = screen.getByLabelText('Line between point 0 and 1');
            fireEvent.pointerMove(line, { ...hoverPoint });

            const ghostPoint = screen.getByLabelText('Add a point between point 0 and 1');
            fireEvent.pointerDown(ghostPoint, { buttons: 1, ...hoverPoint });
            fireEvent.pointerMove(ghostPoint, {
                clientX: hoverPoint.clientX + moveBy.x,
                clientY: hoverPoint.clientY + moveBy.y,
            });
            fireEvent.pointerUp(ghostPoint, { buttons: 1, ...hoverPoint });

            const finalShape = {
                ...mockShape,
                points: [
                    mockShape.points[0],
                    {
                        x: expectedProjectedPoint.x + moveBy.x,
                        y: expectedProjectedPoint.y + moveBy.y,
                    },
                    mockShape.points[1],
                    mockShape.points[2],
                ],
            };
            expect(removeOffLimitPointsPolygon).toHaveBeenCalled();
            expect(mockedUpdateAnnotations).toHaveBeenCalledWith([
                {
                    ...mockAnnotation,
                    shape: finalShape,
                },
            ]);
        });
    });

    describe('removing points from a polygon', () => {
        const mockAnnotation = getMockedAnnotation({
            id: 'polygon-1',
            shape: {
                type: 'polygon',
                points: [
                    { x: 10, y: 10 },
                    { x: 30, y: 10 },
                    { x: 50, y: 30 },
                    { x: 50, y: 50 },
                    { x: 10, y: 50 },
                ],
            },
        }) as Annotation & { shape: Polygon };
        const mockShape = mockAnnotation.shape;

        it('removes a point from a polygon', async () => {
            await renderApp(mockAnnotation);

            const pointToRemove = screen.getByLabelText('Click to select point 0');
            fireEvent.pointerDown(pointToRemove);

            expect(pointToRemove).toHaveAttribute('aria-selected', 'true');
            fireEvent.keyDown(document.body, { key: 'Delete', keyCode: 46, code: 'Delete' });

            const finalShape = {
                ...mockShape,
                points: [mockShape.points[1], mockShape.points[2], mockShape.points[3], mockShape.points[4]],
            };
            expect(removeOffLimitPointsPolygon).toHaveBeenCalledWith(finalShape, mockROI);
            expect(mockedUpdateAnnotations).toHaveBeenCalledWith([
                {
                    ...mockAnnotation,
                    shape: finalShape,
                },
            ]);

            expect(pointToRemove).toHaveAttribute('aria-selected', 'false');
        });

        it('deselects points', async () => {
            await renderApp(mockAnnotation);

            const pointToRemove0 = screen.getByLabelText('Click to select point 0');
            const pointToRemove1 = screen.getByLabelText('Click to select point 1');
            const pointToRemove2 = screen.getByLabelText('Click to select point 2');

            fireEvent.pointerDown(pointToRemove0);
            expect(pointToRemove0).toHaveAttribute('aria-selected', 'true');

            fireEvent.pointerDown(pointToRemove1);
            expect(pointToRemove0).toHaveAttribute('aria-selected', 'false');
            expect(pointToRemove1).toHaveAttribute('aria-selected', 'true');

            fireEvent.pointerDown(pointToRemove0, { shiftKey: true });
            fireEvent.pointerDown(pointToRemove1, { shiftKey: true });
            fireEvent.pointerDown(pointToRemove2, { shiftKey: true });

            expect(pointToRemove0).toHaveAttribute('aria-selected', 'true');
            expect(pointToRemove1).toHaveAttribute('aria-selected', 'false');
            expect(pointToRemove2).toHaveAttribute('aria-selected', 'true');

            expect(pointToRemove0).toHaveAttribute('aria-selected', 'true');
            fireEvent.keyDown(document.body, { key: 'Delete', keyCode: 46, code: 'Delete' });

            const finalShape = {
                ...mockShape,
                points: [mockShape.points[1], mockShape.points[3], mockShape.points[4]],
            };
            expect(removeOffLimitPointsPolygon).toHaveBeenCalledWith(finalShape, mockROI);
            expect(mockedUpdateAnnotations).toHaveBeenCalledWith([
                {
                    ...mockAnnotation,
                    shape: finalShape,
                },
            ]);
        });

        it('removes the polygon if it ends up with 2 points or less', async () => {
            await renderApp(annotation);

            const pointToRemove = screen.getByLabelText('Click to select point 0');
            fireEvent.pointerDown(pointToRemove);
            expect(pointToRemove).toHaveAttribute('aria-selected', 'true');

            const otherPointToRemove = screen.getByLabelText('Shift click to select point 1');
            fireEvent.pointerDown(otherPointToRemove, { shiftKey: true });
            expect(otherPointToRemove).toHaveAttribute('aria-selected', 'true');

            fireEvent.pointerDown(screen.getByLabelText('Shift click to select point 2'), { shiftKey: true });

            fireEvent.keyDown(document.body, { key: 'Delete', keyCode: 46, code: 'Delete' });

            expect(mockedDeleteAnnotations).toBeCalledWith([annotation.id]);
        });
    });
});
