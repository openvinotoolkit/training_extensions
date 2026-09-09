// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { AnnotatedVideoFrame, Label, VideoFramePrediction } from '@/api/types';
import { Flex, View } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';
import { useNumberFormatter } from 'react-aria';

import type { AnnotatorMode } from '../../../../../../../shared/annotator/annotator-mode';
import { EMPTY_LABEL_ID } from '../../../../../../../shared/annotator/labels';
import { useVideoFramesAnnotations } from '../../../../api/use-video-frames-annotations';
import { PREDICTION_CHUNK_SIZE, useVideoFramesPredictions } from '../../../../api/use-video-frames-predictions';
import { useVideoPlayer } from '../../../../video-player-provider.component';

import classes from './video-frame-segment.module.scss';

const getAriaLabel = (label: Label | undefined, isPrediction: boolean) => {
    if (isPrediction) {
        if (label === undefined) {
            return 'No prediction';
        }
        return `Predicted ${label.name}`;
    }

    if (label === undefined) {
        return 'Not labeled';
    }
    return label.name;
};

type LabelSegmentProps = {
    label?: Label;
    striped: boolean;
    isLoading: boolean;
    isPrediction?: boolean;
};

const LabelSegment = ({ label, striped, isLoading, isPrediction = false }: LabelSegmentProps) => {
    if (isLoading) {
        return <View width='100%' height='100%' UNSAFE_className={classes.loadingGradient} />;
    }

    const backgroundColor =
        label !== undefined ? label.color : striped ? 'var(--spectrum-global-color-gray-400)' : undefined;

    return (
        <div aria-label={getAriaLabel(label, isPrediction)} role='presentation' style={{ height: '100%' }}>
            <View
                width='100%'
                height='100%'
                UNSAFE_className={`${classes.labelMarker} ${striped ? classes.stripedBackground : ''}`}
                UNSAFE_style={{ backgroundColor }}
            />
        </div>
    );
};

const SelectedFrameOverlay = () => {
    return (
        <View
            data-testid='selected'
            top={0}
            left={0}
            right={0}
            bottom={0}
            backgroundColor='static-white'
            position='absolute'
            UNSAFE_className={classes.activeFrameOverlay}
        />
    );
};

const selectAnnotationsForFrame = (frameNumber: number) => (data: AnnotatedVideoFrame[]) =>
    data.find(({ frame_index }) => frame_index === frameNumber);

const useVideoTimelineAnnotations = ({ frameNumber }: { frameNumber: number }) => {
    const { step } = useVideoPlayer();
    const { data: videoFramesAnnotations, isPending: isVideoFramesAnnotationsPending } = useVideoFramesAnnotations({
        frameNumber,
        frameSkip: step,
        selector: selectAnnotationsForFrame(frameNumber),
    });

    const annotations = videoFramesAnnotations?.annotation_data?.annotations;

    const annotatedLabels =
        annotations === undefined
            ? undefined
            : isEmpty(annotations)
              ? [EMPTY_LABEL_ID]
              : annotations.flatMap((annotation) => annotation.labels.map(({ id }) => id));

    return {
        annotatedLabels,
        isAnnotationLoading: isVideoFramesAnnotationsPending,
    };
};

const selectPredictionsForFrame = (frameNumber: number) => (data: VideoFramePrediction[]) =>
    data.find(({ media }) => media.frame_index === frameNumber);

const useVideoTimelinePredictions = ({ frameNumber }: { frameNumber: number }) => {
    const { step } = useVideoPlayer();
    const { data, isPending } = useVideoFramesPredictions({
        frameNumber,
        frameSkip: step,
        chunkSize: PREDICTION_CHUNK_SIZE,
        selector: selectPredictionsForFrame(frameNumber),
    });

    const predictedLabels = data?.prediction?.flatMap((prediction) => prediction.labels.map(({ id }) => id));

    return {
        predictedLabels,
        isPredictionLoading: isPending,
    };
};

type LabelSegmentsProps = {
    labels: Label[];
    colIndex: number;
    frameNumber: number;
    renderLabelSegment: (label: Label) => ReactNode;
};

const LabelsSegments = ({ labels, colIndex, frameNumber, renderLabelSegment }: LabelSegmentsProps) => {
    return (
        <Flex gap='size-100' direction='column' marginTop='size-100'>
            {labels.map((label, rowIndex) => {
                return (
                    <div
                        role='gridcell'
                        key={label.id}
                        aria-colindex={colIndex + 1}
                        aria-rowindex={rowIndex + 1}
                        aria-label={`Label ${label.name} in frame number ${frameNumber}`}
                    >
                        <Flex gap='size-10' height='size-225' direction='column' marginEnd={'size-10'}>
                            {renderLabelSegment(label)}
                        </Flex>
                    </div>
                );
            })}
        </Flex>
    );
};

type PredictionsLabelsSegmentsProps = {
    labels: Label[];
    colIndex: number;
    frameNumber: number;
};

const PredictionsLabelsSegments = ({ labels, colIndex, frameNumber }: PredictionsLabelsSegmentsProps) => {
    const { predictedLabels, isPredictionLoading } = useVideoTimelinePredictions({ frameNumber });

    return (
        <LabelsSegments
            labels={labels}
            colIndex={colIndex}
            frameNumber={frameNumber}
            renderLabelSegment={(label) => (
                <LabelSegment
                    isPrediction
                    striped={true}
                    isLoading={isPredictionLoading}
                    label={predictedLabels?.includes(label.id) ? label : undefined}
                />
            )}
        />
    );
};

type AnnotationsLabelsSegmentsProps = {
    labels: Label[];
    colIndex: number;
    frameNumber: number;
};

const AnnotationsLabelsSegments = ({ labels, colIndex, frameNumber }: AnnotationsLabelsSegmentsProps) => {
    const { annotatedLabels, isAnnotationLoading } = useVideoTimelineAnnotations({ frameNumber });

    return (
        <LabelsSegments
            labels={labels}
            colIndex={colIndex}
            frameNumber={frameNumber}
            renderLabelSegment={(label) => {
                return (
                    <LabelSegment
                        striped={false}
                        isLoading={isAnnotationLoading}
                        label={annotatedLabels?.includes(label.id) ? label : undefined}
                    />
                );
            }}
        />
    );
};

type VideoFrameSegmentProps = {
    isFirstFrame: boolean;
    isLastFrame: boolean;
    isSelectedFrame: boolean;
    labels: Label[];
    frameNumber: number;
    showTicks: boolean;
    colIndex: number;
    onClick: (frameNumber: number) => void;
    mode: AnnotatorMode;
};

export const VideoFrameSegment = ({
    isFirstFrame,
    isLastFrame,
    isSelectedFrame,
    showTicks,
    frameNumber,
    labels,
    colIndex,
    onClick,
    mode,
}: VideoFrameSegmentProps) => {
    const formatter = useNumberFormatter({
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
        maximumSignificantDigits: 4,
        notation: 'compact',
    });

    const tickTextStyle = {
        left: isFirstFrame ? '50%' : 'initial',
        right: isLastFrame ? '0%' : 'initial',
    };

    return (
        <div
            onClick={() => {
                onClick(frameNumber);
            }}
            className={classes.videoFrameSegmentContainer}
        >
            <View marginTop={'-0.5rem'} paddingBottom={'0.5rem'} position={'relative'}>
                {showTicks ? (
                    <div className={classes.ticksText} style={tickTextStyle}>
                        {formatter.format(frameNumber)}f
                        <div className={classes.ticksIndicator} />
                    </div>
                ) : (
                    <></>
                )}
            </View>

            {/* TODO: Check if we need this overlay */}
            {/*{isActiveFrame || isFilteredFrame ? <ActiveFrameOverlay /> : <></>}*/}

            {isSelectedFrame && <SelectedFrameOverlay />}

            {mode === 'annotation' ? (
                <AnnotationsLabelsSegments labels={labels} colIndex={colIndex} frameNumber={frameNumber} />
            ) : (
                <PredictionsLabelsSegments labels={labels} colIndex={colIndex} frameNumber={frameNumber} />
            )}
        </div>
    );
};
