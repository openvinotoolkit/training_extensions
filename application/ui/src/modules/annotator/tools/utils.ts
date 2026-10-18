// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type ClipperShape from '@doodle3d/clipper-js';
import Clipper from '@doodle3d/clipper-js';
import type { Shape as SmartToolsShape, Polygon as ToolPolygon, Rect as ToolRect } from '@geti-ui/smart-tools/types';
import { BoundingBox } from '@geti-ui/smart-tools/utils';
import { isEmpty } from 'lodash-es';

import type { ClipperPoint, Point, Polygon, Rect, RegionOfInterest, Shape } from '../../../shared/types';

// @ts-expect-error `default` actually exists in the module
const ClipperJS = Clipper.default || Clipper;

export enum PointerType {
    Mouse = 'mouse',
    Pen = 'pen',
    Touch = 'touch',
}

export const getBoundingBoxInRoi = (boundingBox: RegionOfInterest, roi: RegionOfInterest) => {
    const x = Math.max(0, boundingBox.x);
    const y = Math.max(0, boundingBox.y);

    return {
        x,
        y,
        width: Math.min(roi.width - x, boundingBox.width),
        height: Math.min(roi.height - y, boundingBox.height),
    };
};

export const getBoundingRectFromShape = (shape: Shape): Rect | null => {
    if (shape.type === 'rectangle') {
        return shape;
    }

    if (shape.type === 'full_image') {
        return null;
    }

    const xs = shape.points.map((point: { x: number; y: number }) => point.x);
    const ys = shape.points.map((point: { x: number; y: number }) => point.y);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    const width = Math.max(...xs) - x;
    const height = Math.max(...ys) - y;

    return {
        type: 'rectangle',
        x,
        y,
        width,
        height,
    };
};

export const intersectionOverUnion = (a: Rect, b: Rect): number => {
    const xMin = Math.max(a.x, b.x);
    const yMin = Math.max(a.y, b.y);
    const xMax = Math.min(a.x + a.width, b.x + b.width);
    const yMax = Math.min(a.y + a.height, b.y + b.height);

    const intersectionWidth = Math.max(0, xMax - xMin);
    const intersectionHeight = Math.max(0, yMax - yMin);
    const intersectionArea = intersectionWidth * intersectionHeight;

    if (intersectionArea === 0) {
        return 0;
    }

    const unionArea = a.width * a.height + b.width * b.height - intersectionArea;

    return unionArea === 0 ? 0 : intersectionArea / unionArea;
};

interface getBoundingBoxResizePointsProps {
    gap: number;
    boundingBox: RegionOfInterest;
    onResized: (boundingBox: RegionOfInterest) => void;
}
export const getBoundingBoxResizePoints = ({ boundingBox, gap, onResized }: getBoundingBoxResizePointsProps) => {
    return [
        {
            x: boundingBox.x,
            y: boundingBox.y,
            moveAnchorTo: (x: number, y: number) => {
                const x1 = Math.max(0, Math.min(x, boundingBox.x + boundingBox.width - gap));
                const y1 = Math.max(0, Math.min(y, boundingBox.y + boundingBox.height - gap));

                onResized({
                    x: x1,
                    width: Math.max(gap, boundingBox.width + boundingBox.x - x1),
                    y: y1,
                    height: Math.max(gap, boundingBox.height + boundingBox.y - y1),
                });
            },
            cursor: 'nw-resize',
            label: 'North west resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width / 2,
            y: boundingBox.y,
            moveAnchorTo: (_x: number, y: number) => {
                const y1 = Math.max(0, Math.min(y, boundingBox.y + boundingBox.height - gap));

                onResized({
                    ...boundingBox,
                    y: y1,
                    height: Math.max(gap, boundingBox.height + boundingBox.y - y1),
                });
            },
            cursor: 'n-resize',
            label: 'North resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width,
            y: boundingBox.y,
            moveAnchorTo: (x: number, y: number) => {
                const y1 = Math.max(0, Math.min(y, boundingBox.y + boundingBox.height - gap));

                onResized({
                    ...boundingBox,
                    width: Math.max(gap, x - boundingBox.x),
                    y: y1,
                    height: Math.max(gap, boundingBox.height + boundingBox.y - y1),
                });
            },
            cursor: 'ne-resize',
            label: 'North east resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width,
            y: boundingBox.y + boundingBox.height / 2,
            moveAnchorTo: (x: number) => {
                onResized({ ...boundingBox, width: Math.max(gap, x - boundingBox.x) });
            },
            cursor: 'e-resize',
            label: 'East resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width,
            y: boundingBox.y + boundingBox.height,
            moveAnchorTo: (x: number, y: number) => {
                onResized({
                    x: boundingBox.x,
                    width: Math.max(gap, x - boundingBox.x),

                    y: boundingBox.y,
                    height: Math.max(gap, y - boundingBox.y),
                });
            },
            cursor: 'se-resize',
            label: 'South east resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width / 2,
            y: boundingBox.y + boundingBox.height,
            moveAnchorTo: (_x: number, y: number) => {
                onResized({
                    ...boundingBox,
                    y: boundingBox.y,
                    height: Math.max(gap, y - boundingBox.y),
                });
            },
            cursor: 's-resize',
            label: 'South resize anchor',
        },
        {
            x: boundingBox.x,
            y: boundingBox.y + boundingBox.height,
            moveAnchorTo: (x: number, y: number) => {
                const x1 = Math.max(0, Math.min(x, boundingBox.x + boundingBox.width - gap));

                onResized({
                    x: x1,
                    width: Math.max(gap, boundingBox.width + boundingBox.x - x1),

                    y: boundingBox.y,
                    height: Math.max(gap, y - boundingBox.y),
                });
            },
            cursor: 'sw-resize',
            label: 'South west resize anchor',
        },
        {
            x: boundingBox.x,
            y: boundingBox.y + boundingBox.height / 2,
            moveAnchorTo: (x: number, _y: number) => {
                const x1 = Math.max(0, Math.min(x, boundingBox.x + boundingBox.width - gap));

                onResized({
                    ...boundingBox,
                    x: x1,
                    width: Math.max(gap, boundingBox.width + boundingBox.x - x1),
                });
            },
            cursor: 'w-resize',
            label: 'West resize anchor',
        },
    ];
};

const clampBetween = (min: number, value: number, max: number): number => {
    return Math.max(min, Math.min(max, value));
};

export const getClampedBoundingBox = (point: Point, boundingBox: RegionOfInterest, roi: RegionOfInterest) => {
    const roiX = roi.width + roi.x;
    const roiY = roi.height + roi.y;
    const shapeX = boundingBox.width + boundingBox.x;
    const shapeY = boundingBox.height + boundingBox.y;

    const clampedTranslate = {
        x: clampBetween(shapeX - roiX, -point.x, boundingBox.x - roi.x),
        y: clampBetween(shapeY - roiY, -point.y, boundingBox.y - roi.y),
    };

    return {
        ...boundingBox,
        x: boundingBox.x - clampedTranslate.x,
        y: boundingBox.y - clampedTranslate.y,
    };
};

export function convertToolShapeToGetiShape(shape: ToolPolygon): Polygon;
export function convertToolShapeToGetiShape(shape: ToolRect): Rect;
export function convertToolShapeToGetiShape(shape: SmartToolsShape): Shape;
export function convertToolShapeToGetiShape(shape: SmartToolsShape): Shape {
    switch (shape.shapeType) {
        case 'polygon':
            return { type: 'polygon', points: shape.points };
        case 'rect':
            return {
                type: 'rectangle',
                x: shape.x,
                y: shape.y,
                width: shape.width,
                height: shape.height,
            };
        default:
            throw new Error('Unknown shape type');
    }
}

const removeOffPointsRect = (rect: Rect, roi: RegionOfInterest): Rect => {
    const { x, y, width, height } = roi;

    // Left boundary
    if (rect.x < x) {
        return {
            ...rect,
            x,
            width: rect.width - (x - rect.x),
        };
    }

    // Right boundary
    if (rect.x + rect.width > x + width) {
        const diff = rect.x + rect.width - x - width;
        return {
            ...rect,
            width: rect.width - diff,
        };
    }

    // Top boundary
    if (rect.y < y) {
        return {
            ...rect,
            y,
            height: rect.height - (y - rect.y),
        };
    }

    // Bottom boundary
    if (rect.y + rect.height > y + height) {
        const diff = rect.y + rect.height - y - height;
        return {
            ...rect,
            height: rect.height - diff,
        };
    }

    return rect;
};

export const removeOffLimitPointsPolygon = (shape: Polygon | Rect, roi: RegionOfInterest): Polygon => {
    const { width, height, x, y } = roi;
    const getRect = (rx: number, ry: number, rWidth: number, rHeight: number): Rect => ({
        x: rx,
        y: ry,
        width: rWidth,
        height: rHeight,
        type: 'rectangle',
    });
    // `eraserSize` Builds and positions rect shapes around ROI limits (top, left, right, bottom),
    // finally `getShapesDifference` will use those rects to calc and remove offline polygons
    const eraserSize = 10;
    const topRect = getRect(x - eraserSize, y - eraserSize, width + eraserSize * 3, eraserSize);
    const leftRect = getRect(x - eraserSize, y - eraserSize, eraserSize, height * 2);
    const rightRect = getRect(x + width, y - eraserSize, eraserSize, height * 2);
    const bottomRect = getRect(x - eraserSize, y + height, width + eraserSize * 3, eraserSize);

    return [leftRect, bottomRect, rightRect, topRect].reduce(
        (accum, current) => getShapesDifference(roi, accum, current),
        shape
    ) as Polygon;
};

const calculateRectanglePoints = (shape: BoundingBox): ClipperPoint[] => {
    const { x: X, y: Y, width, height } = shape;
    const topLeftPoint = { X, Y };
    const topRightPoint = { X: X + width, Y };
    const bottomLeftPoint = { X, Y: Y + height };
    const bottomRightPoint = { X: X + width, Y: Y + height };

    return [topLeftPoint, topRightPoint, bottomRightPoint, bottomLeftPoint];
};

const convertPolygonPoints = (shape: Polygon): ClipperPoint[] => {
    return shape.points.map(({ x, y }: Point) => ({ X: x, Y: y }));
};

const transformToClipperShape = (shape: Polygon | Rect): ClipperShape => {
    if (shape.type === 'rectangle') {
        return new ClipperJS([calculateRectanglePoints(shape)], true);
    } else {
        return new ClipperJS([convertPolygonPoints(shape)], true);
    }
};

const runUnionOrDifference =
    <T>(algorithm: 'union' | 'difference', formatTo: (path: ClipperPoint[]) => T) =>
    (roi: RegionOfInterest, subj: Polygon | Rect, clip: Polygon | Rect): T => {
        const subjShape = transformToClipperShape(subj);
        const clipShape = transformToClipperShape(clip);
        const solutionPath = subjShape[algorithm](clipShape);
        const filteredPath = filterIntersectedPathsWithRoi(roi, solutionPath);
        const biggestPath = findBiggerSubPath(filteredPath);

        return formatTo(biggestPath);
    };

const clipperShapeToPolygon = (path: ClipperPoint[]): Polygon => ({
    type: 'polygon',
    points: path.map(({ X, Y }) => ({ x: X, y: Y })),
});

const getShapesDifference = runUnionOrDifference<Polygon>('difference', clipperShapeToPolygon);

const findBiggerSubPath = (shape: ClipperShape): ClipperPoint[] => {
    const areas = shape.areas();
    const { index: shapeIndex } = areas.reduce(
        (accum: { value: number; index: number }, value, index) => {
            return value > accum.value ? { value, index } : accum;
        },
        { value: 0, index: 0 }
    );

    return shape.paths.length ? shape.paths[shapeIndex] : [];
};

const hasIntersection = (clip: ClipperShape, subj: ClipperShape) => {
    const { paths } = clip.intersect(subj);

    return !isEmpty(paths);
};

const filterIntersectedPathsWithRoi = (roi: RegionOfInterest, shape: ClipperShape): ClipperShape => {
    const newPath = shape.clone();
    const roiRect = transformToClipperShape({ ...roi, type: 'rectangle' });

    newPath.paths = newPath.paths.filter((subPath) => hasIntersection(roiRect, new ClipperJS([subPath])));

    return newPath;
};

export const removeOffLimitPoints = (shape: Shape, roi: RegionOfInterest): Shape => {
    return shape.type === 'rectangle'
        ? removeOffPointsRect(shape, roi)
        : removeOffLimitPointsPolygon(shape as Polygon, roi);
};

type ElementType = SVGElement | HTMLDivElement;
export const getRelativePoint = (element: ElementType, point: Point, zoom: number): Point => {
    const rect = element.getBoundingClientRect();

    return {
        x: Math.round((point.x - rect.left) / zoom),
        y: Math.round((point.y - rect.top) / zoom),
    };
};

export const loadImage = (link: string): Promise<HTMLImageElement> =>
    new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';

        image.onload = () => resolve(image);
        image.onerror = (error) => reject(error);

        image.fetchPriority = 'high';
        image.src = link;

        if (process.env.NODE_ENV === 'test') {
            // Immediately load the media item's image
            resolve(image);
        }
    });

const drawImageOnCanvas = (img: HTMLImageElement, filter = ''): HTMLCanvasElement => {
    const canvas: HTMLCanvasElement = document.createElement('canvas');

    canvas.width = img.naturalWidth ? img.naturalWidth : img.width;
    canvas.height = img.naturalHeight ? img.naturalHeight : img.height;

    const ctx = canvas.getContext('2d');

    if (ctx) {
        ctx.filter = filter;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    return canvas;
};

// Canvas rasterization limits (Chrome: 16384² ≈ 268 Mpx total area).
// Check dimensions upfront: some browsers return a valid 2D context for
// oversized canvases but then fail/throw at getImageData time (returning
// blank buffers or OOM), which caused the original "white square" issue.
const MAX_CANVAS_SIDE = 16384;

export const canRasteriseAtFullSize = (width: number, height: number): boolean =>
    width <= MAX_CANVAS_SIDE && height <= MAX_CANVAS_SIDE;

export const getImageData = (img: HTMLImageElement): ImageData => {
    if (img.width === 0 && img.height === 0) return new ImageData(1, 1);

    const sourceWidth = img.naturalWidth || img.width;
    const sourceHeight = img.naturalHeight || img.height;

    // Oversized media cannot be rasterised to a canvas (and smart tools can't
    // run on it either), so return a placeholder; it is displayed via <img>.
    if (!canRasteriseAtFullSize(sourceWidth, sourceHeight)) {
        return new ImageData(1, 1);
    }

    const canvas = drawImageOnCanvas(img);
    const ctx = canvas.getContext('2d');

    return ctx !== null ? ctx.getImageData(0, 0, canvas.width, canvas.height) : new ImageData(1, 1);
};

export const isKeyboardDelete = (event: KeyboardEvent): boolean =>
    event.code === 'Backspace' || event.code === 'Delete';

type ProjectLine = [startPoint: Point, endPoint: Point];
export const projectPointOnLine = ([startPoint, endPoint]: ProjectLine, point: Point): Point | undefined => {
    // Move startPoint to origin
    const b = {
        x: endPoint.x - startPoint.x,
        y: endPoint.y - startPoint.y,
    };
    const a = {
        x: point.x - startPoint.x,
        y: point.y - startPoint.y,
    };

    // Project a onto b
    const aDotB = a.x * b.x + a.y * b.y;
    const bDotB = b.x * b.x + b.y * b.y;
    const scale = aDotB / bDotB;

    // Return undefined if the projected point would lie outside of the given line
    if (scale < 0 || scale > 1) {
        return undefined;
    }

    // Move origin back to startPoint
    return {
        x: b.x * scale + startPoint.x,
        y: b.y * scale + startPoint.y,
    };
};

/**
 * Oversized image handling (Chrome canvas limit ~268 Mpx):
 * - `getImageData` returns a 1×1 placeholder for media too large to rasterise.
 * - Such media is displayed via an <img> tag (see annotator-canvas.tsx).
 * - Smart tools (SAM, magnetic lasso, SSIM) are disabled for it, because they
 *   rely on a full-resolution canvas/OpenCV buffer that cannot be produced.
 */

/**
 * Checks if an ImageData is a placeholder for oversized media.
 * Full-size ImageData satisfies width * height * 4 entries in `data`
 * (RGBA: 4 channels, 1 byte each). A 1×1 placeholder breaks that.
 * @returns true if data doesn't match full size (i.e., is a placeholder)
 */
export const isImageOversized = (image: ImageData): boolean => {
    return image.data.length !== image.width * image.height * 4;
};
