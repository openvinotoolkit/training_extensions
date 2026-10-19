// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

import type { Label } from '@/api/types';
import { useSizeHook } from 'hooks/use-size.hook';

import type { AnnotatorMode } from '../../../../annotator-mode';
import { useVideoPlayer } from '../../../video-player-provider.component';
import { useHorizontalVirtualizer } from './use-horizontal-virtualizer.hook';
import { VideoFrameSegments } from './video-frame-segment/video-frame-segments.component';
import { VideoPlayerSlider } from './video-player-slider/video-player-slider.component';

import classes from './video-timeline.module.scss';

type VideoTimelineProps = {
    labels: Label[];
    mode: AnnotatorMode;
};

const MIN_SIZE_OF_SEGMENT = 2 * 8;

export const VideoTimeline = ({ labels, mode }: VideoTimelineProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const size = useSizeHook(containerRef);
    const { videoFrame, videoControls, step } = useVideoPlayer();
    const { isPlaying } = videoControls;

    const frameNumber = videoFrame.frame_number;
    const totalFrames = videoFrame.frame_count;

    const totalSegments = Math.max(Math.ceil(totalFrames / step), 0);
    const sizePerSquare = size === undefined ? 0 : Math.max(MIN_SIZE_OF_SEGMENT, size.width / totalSegments);
    const frameOffset = Math.round(sizePerSquare / 2);

    const { outerRef, items, innerStyle, scrollToItem } = useHorizontalVirtualizer({
        itemCount: totalSegments,
        itemSize: sizePerSquare,
        viewportWidth: size?.width ?? 0,
        overscanCount: 20,
    });

    useEffect(() => {
        scrollToItem(Math.round(frameNumber / step));
    }, [scrollToItem, frameNumber, step]);

    return (
        <div ref={containerRef}>
            <div ref={outerRef} style={{ overflow: 'auto', width: size?.width }}>
                <div style={{ width: sizePerSquare * totalSegments }} className={classes.timelineSliderWrapper}>
                    <VideoPlayerSlider
                        ref={outerRef}
                        videoFrame={videoFrame}
                        step={step}
                        frameNumber={frameNumber}
                        sizePerSquare={sizePerSquare}
                        frameOffset={frameOffset}
                        selectFrame={videoControls.goto}
                    />
                </div>
                <VideoFrameSegments
                    mode={mode}
                    frameNumber={frameNumber}
                    style={innerStyle}
                    items={items}
                    containerSize={size?.width ?? 0}
                    labels={labels}
                    step={step}
                    totalSegments={totalSegments}
                    minSizeOfSegment={MIN_SIZE_OF_SEGMENT}
                    isPlaying={isPlaying}
                    selectFrame={videoControls.goto}
                />
            </div>
        </div>
    );
};
