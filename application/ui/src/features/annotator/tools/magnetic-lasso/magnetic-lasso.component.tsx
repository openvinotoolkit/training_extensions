// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState, useTransition } from 'react';

import { toast } from '@/components/toast/toast.component';
import { useZoom } from '@/components/zoom/zoom.provider';
import { isPointOverPoint, isPolygonValid } from '@geti-ui/smart-tools/utils';
import { useMutation } from '@tanstack/react-query';
import { isEmpty, isEqual, throttle } from 'lodash-es';

import { Point } from '../../../../shared/types';
import { isNonEmptyArray } from '../../../../shared/util';
import { useAnnotatorLabels } from '../../annotator-labels-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { usePolygonConfig } from '../hooks/use-polygon-config.hook';
import { PolygonDraw } from '../polygon-tool/polygon-draw.component';
import {
    drawingStyles,
    getToolIcon,
    isCloseMode,
    isPolygonReadyToClose,
    leftRightMouseButtonHandler,
    PolygonMode,
    START_POINT_FIELD_DEFAULT_RADIUS,
    START_POINT_FIELD_FOCUS_RADIUS,
} from '../polygon-tool/utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';

import classes from './magnetic-lasso.module.scss';

export const MagneticLasso = () => {
    const { scale: zoom } = useZoom();
    const [mode, setMode] = useState<PolygonMode>(PolygonMode.MagneticLasso);
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { image } = useSelectedMediaItem();
    const { selectedLabel } = useAnnotatorLabels();
    const [isPendingPolygonOptimization, startTransition] = useTransition();

    const canvasRef = useRef<SVGRectElement>({} as SVGRectElement);
    const isLoading = useRef<boolean>(false);
    const isPointerDown = useRef<boolean>(false);
    const isFreeDrawing = useRef<boolean>(false);
    const buildMapPoint = useRef<Point | null>(null);
    const pendingHistoryCommit = useRef<boolean>(false);

    const {
        worker,
        polygon,
        setPointerLine,
        lassoSegment,
        setLassoSegment,
        isMounted,
        segments,
        setSegments,
        resetTool,
        optimizePolygonOrSegments,
        onPointerMoveRemove,
        setPointFromEvent,
    } = usePolygonConfig({ image, zoom, canvasRef });

    const toolIcon = getToolIcon(mode);

    const STARTING_POINT_RADIUS = Math.ceil(
        (isCloseMode(mode) ? START_POINT_FIELD_FOCUS_RADIUS : START_POINT_FIELD_DEFAULT_RADIUS) / zoom
    );

    useEffect(() => {
        updateBuildMapAfterUndoRedo();

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [mode, segments]);

    const updateBuildMapAfterUndoRedo = (): void => {
        if (mode !== PolygonMode.MagneticLasso || isPointerDown.current) return;

        if (isEmpty(segments)) {
            isLoading.current = false;
            worker?.cleanPoints();

            return;
        }

        const lastSegment = segments.at(-1);
        const lastPoint = Array.isArray(lastSegment) ? lastSegment.at(-1) : undefined;
        const currentBuildMapPoint = buildMapPoint.current ?? { x: NaN, y: NaN };

        if (lastPoint && !isEqual(currentBuildMapPoint, lastPoint)) {
            worker?.buildMap(lastPoint);
        }
    };

    const complete = async () => {
        if (isPolygonReadyToClose(polygon)) {
            startTransition(async () => {
                const optimizedPolygon = await optimizePolygonOrSegments(polygon);

                addAndSelectAnnotations(
                    [{ type: 'polygon', points: optimizedPolygon.points }],
                    selectedLabel ? [selectedLabel] : []
                );
            });
        }
    };

    const canPathBeClosed = (point: Point): boolean => {
        const flatSegments = segments.flat();

        if (isEmpty(flatSegments)) return false;

        return (
            Boolean(polygon) &&
            isPolygonValid({ shapeType: 'polygon', points: polygon?.points ?? [] }) &&
            isPointOverPoint(flatSegments[0], point, STARTING_POINT_RADIUS)
        );
    };

    const handleIsStartingPointHovered = (point: Point): void => {
        if (!isCloseMode(mode) && canPathBeClosed(point)) {
            setMode(PolygonMode.MagneticLassoClose);
        }

        if (isCloseMode(mode) && !canPathBeClosed(point)) {
            setMode(PolygonMode.MagneticLasso);
        }
    };

    const isFreeDrawingAndPathCannotBeClosed = (canBeClosed: boolean): boolean => isFreeDrawing.current && !canBeClosed;

    const addFirstPointOrNewOne = (point: Point): ((prevSegments: Point[][]) => Point[][]) | Point[][] => {
        const hasFirstPoint = !isEmpty(segments.at(-1));

        return hasFirstPoint ? (prevSegments: Point[][]) => [...prevSegments, lassoSegment] : [[point]];
    };

    const getSegmentMutation = useMutation({
        mutationFn: async (point: Point) => worker?.calcPoints(point),

        onError: (): void => {
            toast({ message: 'Failed to select the shape boundaries, could you please try again?', type: 'error' });
        },

        onSuccess: (newPoints?: Point[]) => {
            if (isMounted && isNonEmptyArray(newPoints)) {
                setLassoSegment(newPoints);
                setPointerLine(() => [...segments.flat(), ...lassoSegment]);
            }
        },

        onSettled: () => {
            isLoading.current = false;
        },
    });

    const setBuildMapAndSegment = (point: Point) => {
        setLassoSegment([]);
        setSegments(addFirstPointOrNewOne(point), true);

        isLoading.current = true;
        buildMapPoint.current = point;
        pendingHistoryCommit.current = true;
        worker?.buildMap(point);
    };

    const onPointerDown = leftRightMouseButtonHandler(
        setPointFromEvent((point: Point): void => {
            if (isLoading.current) {
                return;
            }

            isPointerDown.current = true;

            if (!buildMapPoint.current || !isEqual(buildMapPoint.current, point)) {
                setBuildMapAndSegment(point);
            }
        }),
        () => {
            setMode(PolygonMode.Eraser);

            isLoading.current = false;
            buildMapPoint.current = null;
            worker?.cleanPoints();
        }
    );

    const onPointerMove = throttle(
        setPointFromEvent((point: Point): void => {
            if (isEmpty(segments)) return;

            handleIsStartingPointHovered(point);

            if (!isPointerDown.current) {
                return getSegmentMutation.mutate(point);
            }

            isFreeDrawing.current = true;
            buildMapPoint.current = null;
            isLoading.current = false;
            worker?.cleanPoints();

            setLassoSegment((newLassoSegment: Point[]) => [...newLassoSegment, point]);
            setPointerLine(() => [...segments.flat(), ...lassoSegment]);
        }),
        250
    );

    const onPointerUp = setPointFromEvent((point: Point): void => {
        const canBeClosed = canPathBeClosed(point);

        if (isFreeDrawingAndPathCannotBeClosed(canBeClosed)) {
            setBuildMapAndSegment(point);
        }

        if (canBeClosed) {
            complete();
            resetTool();

            isLoading.current = false;
            worker?.cleanPoints();
            pendingHistoryCommit.current = false;
        } else if (pendingHistoryCommit.current) {
            setSegments((prev) => prev);
            pendingHistoryCommit.current = false;
        }

        setMode(PolygonMode.MagneticLasso);
        isPointerDown.current = false;
        isFreeDrawing.current = false;
    });

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={canvasRef}
            aria-label={'polygon tool'}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            onPointerMove={mode === PolygonMode.Eraser ? onPointerMoveRemove : onPointerMove}
            onPointerLeave={mode === PolygonMode.Eraser ? onPointerMoveRemove : onPointerMove}
            style={{ cursor: `url(${toolIcon.cursorUrl}) ${toolIcon.offset}, auto` }}
        >
            {polygon !== null && !isEmpty(polygon.points) && (
                <PolygonDraw
                    shape={polygon}
                    ariaLabel='new polygon'
                    styles={drawingStyles(selectedLabel)}
                    className={isPendingPolygonOptimization ? classes.inputTool : ''}
                    indicatorRadius={STARTING_POINT_RADIUS}
                />
            )}
        </SvgToolCanvas>
    );
};
