// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, RefObject, useEffect, useRef, useState } from 'react';

import type { MediaVideoFrame } from '@/api/types';
import { defer } from 'lodash-es';
import { useHover } from 'react-aria';
import { useDebounceCallback } from 'usehooks-ts';

import { FRAME_STEP_TO_DISPLAY_ALL_FRAMES } from '../../../frame-step/utils';
import { ThumbnailPreview } from './thumbnail-preview.component';
import { VideoSlider } from './video-slider.component';

import classes from './video-slider.module.scss';

type VideoPlayerSliderProps = {
    videoFrame: MediaVideoFrame;
    step: number;
    frameNumber: number;
    sizePerSquare?: number;
    frameOffset?: number;
    ref?: RefObject<HTMLDivElement | null>;
    selectFrame: (frameNumber: number) => void;
};

const THUMBNAIL_DELAY = 1000;

const useShowThumbnail = () => {
    const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    const [thumbnailVideoFrame, setThumbnailVideoFrame] = useState<null | number>(null);
    const [showThumbnail, setShowThumbnail] = useState(false);
    const [thumbnailPosition, setThumbnailPosition] = useState<null | number>(null);
    const setThumbnailVideoFrameDebounced = useDebounceCallback(setThumbnailVideoFrame, 200);

    const { hoverProps } = useHover({
        onHoverStart: () => {
            timeoutRef.current = setTimeout(() => setShowThumbnail(true), THUMBNAIL_DELAY);
        },
        onHoverEnd: () => {
            timeoutRef.current && clearTimeout(timeoutRef.current);
            setShowThumbnail(false);
            setThumbnailPosition(null);
            setThumbnailVideoFrameDebounced(null);
        },
    });

    useEffect(() => {
        return () => {
            timeoutRef.current && clearTimeout(timeoutRef.current);
        };
    }, []);

    return {
        hoverProps,
        onShowThumbnail: setShowThumbnail,
        showThumbnail,
        thumbnailVideoFrame,
        thumbnailPosition,
        onSetThumbnailPosition: setThumbnailPosition,
        onSetThumbnailVideoFrameDebounced: setThumbnailVideoFrameDebounced,
    };
};

const getFrameNumber = (x: number, width: number, minValue: number, maxValue: number, step: number) => {
    const frameNumber = Math.min(
        Math.floor(maxValue / step) * step,
        Math.max(0, Math.round(Math.round((x / width) * (maxValue - minValue)) / step) * step)
    );

    return frameNumber;
};

const getContainerScroll = (ref?: RefObject<HTMLDivElement | null>) => {
    if (ref === undefined) {
        return 0;
    }

    if (ref.current) {
        return ref.current.scrollLeft;
    }

    return 0;
};

const blurActiveInput = (isFocused: boolean): void => {
    const element = document.activeElement;

    if (isFocused && element?.nodeName === 'INPUT') {
        defer(() => (element as HTMLInputElement).blur());
    }
};

export const VideoPlayerSlider = ({
    ref,
    videoFrame,
    step,
    frameNumber,
    sizePerSquare,
    selectFrame,
    frameOffset = 0,
}: VideoPlayerSliderProps) => {
    const [dragFrameNumber, setDragFrameNumber] = useState<number | null>(null);
    const sliderValue = dragFrameNumber ?? frameNumber;

    const {
        hoverProps,
        showThumbnail,
        onShowThumbnail,
        onSetThumbnailPosition,
        onSetThumbnailVideoFrameDebounced,
        thumbnailVideoFrame,
        thumbnailPosition,
    } = useShowThumbnail();

    const framesCount = videoFrame.frame_count;
    const minValue = 0;
    const maxValue = framesCount - 1;
    const isDisplayingAllFrames = FRAME_STEP_TO_DISPLAY_ALL_FRAMES === step;
    const lastFrame = isDisplayingAllFrames ? maxValue : framesCount - step;
    const isLastFrame = sliderValue >= lastFrame;
    const containerScrollLeft = getContainerScroll(ref);

    const handlePointerMove = (event: PointerEvent<HTMLDivElement>): void => {
        const rect = event.currentTarget.getBoundingClientRect();

        const thumbnailPosX = Math.max(0, event.clientX - rect.x) - containerScrollLeft;

        const thumbnailFrameNumber = getFrameNumber(
            thumbnailPosX - frameOffset + containerScrollLeft,
            rect.width,
            minValue,
            maxValue,
            step
        );

        onSetThumbnailPosition(thumbnailPosX);
        onSetThumbnailVideoFrameDebounced(thumbnailFrameNumber);
    };

    return (
        <div className={classes.sliderWrapper} onPointerMove={handlePointerMove} {...hoverProps}>
            <VideoSlider
                isFilled
                width={'100%'}
                aria-label={'Video timeline'}
                showValueLabel={false}
                minValue={minValue}
                maxValue={maxValue}
                value={sliderValue}
                onChange={(newFrameNumber) => {
                    setDragFrameNumber(newFrameNumber);
                    onShowThumbnail(true);
                }}
                onChangeEnd={(newFrameNumber) => {
                    selectFrame(newFrameNumber);
                    setDragFrameNumber(null);
                    blurActiveInput(true);
                }}
                step={step}
                isLastFrame={isLastFrame}
                sizePerSquare={sizePerSquare}
                leftOffset={frameOffset}
            />
            {showThumbnail && thumbnailVideoFrame !== null && thumbnailPosition !== null && (
                <ThumbnailPreview
                    x={thumbnailPosition}
                    width={100}
                    height={100}
                    frameNumber={thumbnailVideoFrame}
                    videoFrame={videoFrame}
                />
            )}
        </div>
    );
};
