// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { VideoFramePrediction } from '@/api/types';
import { usePrefetchQuery, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { mediaPredictionsQueryOptions } from '../../api/use-media-predictions';
import { usePredictionSetup } from '../../predictions-setup-provider.component';
import { useVideoPlayer } from '../video-player-provider.component';
import { getVideoFrameRangeIndexes } from './utils';

export const PREDICTION_CHUNK_SIZE = 15;
export const PREDICTION_FRAME_SKIP = 1;

const useVideoFramesPredictionsQueryOptions = ({
    frameNumber,
    frameSkip,
    rangeStride,
    chunkSize,
}: {
    frameNumber: number;
    frameSkip: number;
    rangeStride?: number;
    chunkSize?: number;
}) => {
    const projectId = useProjectIdentifier();
    const { videoFrame } = useVideoPlayer();
    const { selectedModel, selectedDevice, confidenceThreshold } = usePredictionSetup();

    const { startFrameIndex, endFrameIndex } = getVideoFrameRangeIndexes({
        frames: videoFrame.frame_count - 1,
        frameSkip,
        frameNumber,
        chunkSize,
    });

    return mediaPredictionsQueryOptions({
        projectId,
        selectedModel,
        device: selectedDevice,
        confidenceThreshold,
        mediaId: videoFrame.id,
        range: { stride: rangeStride ?? frameSkip, start_frame: startFrameIndex, end_frame: endFrameIndex },
    });
};

export const usePrefetchVideoFramesPredictions = ({
    frameNumber,
    frameSkip,
    rangeStride,
    chunkSize,
}: {
    frameNumber: number;
    frameSkip: number;
    rangeStride?: number;
    chunkSize?: number;
}) => {
    const queryOptions = useVideoFramesPredictionsQueryOptions({
        frameSkip,
        frameNumber,
        chunkSize,
        rangeStride,
    });

    return usePrefetchQuery(queryOptions);
};

export const useKeepVideoFramesPredictionsSubscribed = ({
    frameNumber,
    frameSkip,
    rangeStride,
    chunkSize,
}: {
    frameNumber: number;
    frameSkip: number;
    rangeStride?: number;
    chunkSize?: number;
}) => {
    const queryOptions = useVideoFramesPredictionsQueryOptions({ frameSkip, frameNumber, chunkSize, rangeStride });

    useQuery({
        ...queryOptions,
        notifyOnChangeProps: [],
        staleTime: Infinity,
        refetchOnMount: false,
    });
};

export const useVideoFramesPredictions = <T>({
    frameNumber,
    frameSkip,
    selector,
    rangeStride,
    chunkSize,
}: {
    frameNumber: number;
    frameSkip: number;
    rangeStride?: number;
    selector: (data: VideoFramePrediction[]) => T;
    chunkSize?: number;
}) => {
    const queryOptions = useVideoFramesPredictionsQueryOptions({
        frameSkip,
        frameNumber,
        chunkSize,
        rangeStride,
    });

    return useQuery({ ...queryOptions, select: selector, refetchOnMount: false, staleTime: Infinity });
};
