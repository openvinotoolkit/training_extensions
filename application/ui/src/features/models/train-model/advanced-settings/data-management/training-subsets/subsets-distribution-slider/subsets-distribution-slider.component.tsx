// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useRef } from 'react';

import { useTranslation } from '@/i18n';
import { Content, ContextualHelp, Flex, Text, View, VisuallyHidden } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { mergeProps, useFocusRing, useNumberFormatter, useSlider, useSliderThumb } from 'react-aria';
import { SliderState, useSliderState } from 'react-stately';

import classes from './subsets-distribution-slider.module.scss';

type ThumbProps = {
    index: number;
    state: SliderState;
    trackRef: RefObject<HTMLDivElement | null>;
    ariaLabel: string;
};

const Thumb = ({ state, trackRef, index, ariaLabel }: ThumbProps) => {
    const inputRef = useRef(null);
    const { thumbProps, inputProps } = useSliderThumb(
        {
            index,
            trackRef,
            inputRef,
        },
        state
    );

    const { focusProps } = useFocusRing();

    return (
        <div
            {...thumbProps}
            className={classes.thumb}
            style={{
                left: `${state.getThumbPercent(index) * 100}%`,
            }}
            aria-label={ariaLabel}
        >
            <VisuallyHidden>
                <input ref={inputRef} {...mergeProps(inputProps, focusProps)} />
            </VisuallyHidden>
        </div>
    );
};

const DistributionTooltip = () => {
    const { t } = useTranslation();

    return (
        <ContextualHelp variant='info'>
            <Content>
                <Text>{t('models.training.dataManagement.trainingSubsets.description')}</Text>
            </Content>
        </ContextualHelp>
    );
};

type SubsetsDistributionSliderProps = {
    label: string;
    onChangeEnd: (values: number[] | number) => void;
    value: number | number[];
    defaultValue?: number | number[];
    onChange: (values: number[] | number) => void;
    maxValue: number;
    step: number;
    minValue: number;
    formatOptions?: Intl.NumberFormatOptions;
};

export const SubsetsDistributionSlider = (props: SubsetsDistributionSliderProps) => {
    const trackRef = useRef(null);

    const numberFormatter = useNumberFormatter(props.formatOptions);
    const state = useSliderState({ ...props, numberFormatter });
    const { trackProps, labelProps } = useSlider(props, state, trackRef);

    const trainingValue = parseInt(state.getThumbValueLabel(0));
    const validationValue = parseInt(state.getThumbValueLabel(1)) - parseInt(state.getThumbValueLabel(0));
    const testValue = props.maxValue - trainingValue - validationValue;

    return (
        <>
            <View gridArea={'label'}>
                <label {...labelProps}>
                    {props.label} <DistributionTooltip />
                </label>
            </View>
            <Flex gridArea={'slider'} alignItems={'center'} gap={'size-150'}>
                <div {...trackProps} ref={trackRef} className={classes.trackContainer}>
                    <View
                        width={`${state.getThumbPercent(0) * 100}%`}
                        UNSAFE_className={clsx(classes.track, classes.trainingTrack)}
                    />
                    <View
                        width={`${(state.getThumbPercent(1) - state.getThumbPercent(0)) * 100}%`}
                        left={`${state.getThumbPercent(0) * 100}%`}
                        UNSAFE_className={clsx(classes.track, classes.validationTrack)}
                    />
                    <View
                        width={`${100 - state.getThumbPercent(1) * 100}%`}
                        left={`${state.getThumbPercent(1) * 100}%`}
                        UNSAFE_className={clsx(classes.track, classes.testTrack)}
                    />
                    <Thumb index={0} state={state} trackRef={trackRef} ariaLabel={'Start range'} />
                    <Thumb index={1} state={state} trackRef={trackRef} ariaLabel={'End range'} />
                </div>
                <Text width={'size-1000'}>
                    <span aria-label={'Training subsets distribution'}>
                        {trainingValue}/{validationValue}/{testValue}%
                    </span>
                </Text>
            </Flex>
        </>
    );
};
