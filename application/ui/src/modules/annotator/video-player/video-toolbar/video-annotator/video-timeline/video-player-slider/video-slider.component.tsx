// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentRef, RefObject, useRef } from 'react';

import { useNumberFormatter, VisuallyHidden, type SpectrumSliderProps } from '@geti-ui/ui';
import { AriaSliderProps, mergeProps, useFocusRing, useSlider, useSliderThumb } from 'react-aria';
import { SliderState, useSliderState } from 'react-stately';

import classes from './video-slider.module.scss';

type ThumbProps = {
    index: number;
    state: SliderState;
    trackWidth: string;
    trackRef: RefObject<HTMLDivElement | null>;
};

const Thumb = ({ state, trackRef, index, trackWidth }: ThumbProps) => {
    const inputRef = useRef(null);
    const { focusProps } = useFocusRing();
    const { thumbProps, inputProps } = useSliderThumb(
        {
            index,
            trackRef,
            inputRef,
        },
        state
    );

    return (
        <div {...thumbProps} style={{ ...thumbProps.style, left: trackWidth }} className={classes.thumb}>
            <VisuallyHidden>
                <input ref={inputRef} {...mergeProps(inputProps, focusProps)} />
            </VisuallyHidden>
        </div>
    );
};

type VideoSliderProps = SpectrumSliderProps & {
    isLastFrame?: boolean;
    sizePerSquare?: number;
    leftOffset?: number;
};

const THUMB_INDEX = 0;

const getTrackWidthWhileIsLastFrame = (trackRef: RefObject<HTMLDivElement | null>, leftOffset: number) => {
    if (trackRef.current === null) {
        return `0px`;
    }

    const THUMB_OFFSET = 8;

    const width = trackRef.current.getBoundingClientRect().width - THUMB_OFFSET - leftOffset;

    return `${width}px`;
};

export const VideoSlider = ({ leftOffset = 0, sizePerSquare, isLastFrame, ...props }: VideoSliderProps) => {
    const trackRef = useRef<ComponentRef<'div'>>(null);
    const sliderProps = props as unknown as AriaSliderProps<number | number[]>;

    const numberFormatter = useNumberFormatter(props.formatOptions);
    const state = useSliderState({ ...props, numberFormatter });
    const [value] = state.values;

    const { groupProps, trackProps } = useSlider(sliderProps, state, trackRef);

    const trackWidth = isLastFrame
        ? getTrackWidthWhileIsLastFrame(trackRef, leftOffset)
        : sizePerSquare
          ? `${sizePerSquare * (value / state.step)}px`
          : `${state.getThumbPercent(THUMB_INDEX) * 100}%`;

    return (
        <div
            {...groupProps}
            className={classes.slider}
            style={{
                overflow: leftOffset === 0 || value === props.minValue ? 'visible' : 'hidden',
            }}
        >
            <div {...trackProps} ref={trackRef} className={classes.track} style={{ left: `${leftOffset}px` }}>
                <div className={classes.lowerTrack} style={{ width: trackWidth }}></div>

                <Thumb index={THUMB_INDEX} state={state} trackRef={trackRef} trackWidth={trackWidth} />
            </div>
        </div>
    );
};
