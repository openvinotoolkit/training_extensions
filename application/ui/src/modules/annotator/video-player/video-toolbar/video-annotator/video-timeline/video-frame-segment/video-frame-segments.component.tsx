// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties } from 'react';

import type { Label } from '@/api/types';
import { Flex } from '@geti-ui/ui';

import type { AnnotatorMode } from '../../../../../annotator-mode';
import type { VirtualItem } from '../use-horizontal-virtualizer.hook';
import { VideoFrameSegment } from './video-frame-segment.component';

import classes from './video-frame-segment.module.scss';

const FRAMES_BEFORE_NEXT_TICK = 6;

type VideoFrameSegmentsProps = {
    labels: Label[];
    totalSegments: number;
    step: number;
    isPlaying: boolean;
    minSizeOfSegment: number;
    containerSize: number;
    style?: CSSProperties;
    items: VirtualItem[];
    frameNumber: number;
    selectFrame: (frameNumber: number) => void;
    mode: AnnotatorMode;
};

export const VideoFrameSegments = ({
    mode,
    totalSegments,
    labels,
    step,
    isPlaying,
    minSizeOfSegment,
    containerSize,
    style,
    items,
    frameNumber,
    selectFrame,
}: VideoFrameSegmentsProps) => {
    return (
        <div
            className={classes.videoFrameSegmentsContainer}
            style={{ ...style, gridTemplateColumns: `repeat(auto-fit, minmax(${minSizeOfSegment}px, max-content))` }}
            role='grid'
            aria-label={'Video timeline'}
            aria-colcount={totalSegments}
            aria-rowcount={labels.length}
        >
            {items.map(({ index, size: width = 30 }) => {
                const itemFrameNumber = index * step;

                const lastFrame = totalSegments - 1;
                const isLastFrame = index === lastFrame;
                const isFirstFrame = index === 0;
                const isHalfFrame =
                    index + FRAMES_BEFORE_NEXT_TICK < lastFrame &&
                    index % Math.ceil((0.5 * containerSize) / width) === 0;

                const isSelectedFrame = isPlaying === false && itemFrameNumber === frameNumber;

                // Always show ticks at the start and end of the video or at every frame
                // that is on half of the timeline
                const showTicks = isFirstFrame || isHalfFrame || isLastFrame;

                return (
                    <Flex justifyContent='center' alignItems='center' width={width} key={itemFrameNumber}>
                        <VideoFrameSegment
                            mode={mode}
                            colIndex={index}
                            onClick={selectFrame}
                            labels={labels}
                            showTicks={showTicks}
                            frameNumber={itemFrameNumber}
                            isSelectedFrame={isSelectedFrame}
                            isLastFrame={isLastFrame}
                            isFirstFrame={isFirstFrame}
                        />
                    </Flex>
                );
            })}
        </div>
    );
};
