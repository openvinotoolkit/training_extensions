// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fetchClient } from '@/api';
import type { PredictionDTO, PredictionVideoRangePayload } from '@/api/types';
import { queryOptions, useIsFetching, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { getModelIdentifierPayload, SelectableModel } from '../../models/utils';
import { usePredictionSetup } from '../predictions-setup-provider.component';
import { useSelectedMediaItem } from '../selected-media-item-provider.component';
import { PREDICTION_CHUNK_SIZE, PREDICTION_FRAME_SKIP } from '../video-player/api/use-video-frames-predictions';
import { getVideoFrameRangeIndexes } from '../video-player/api/utils';
import { useVideoPlayerContext } from '../video-player/video-player-provider.component';

// Prefix matching every prediction query of a project, so callers can cancel them all at once.
export const getMediaPredictionsQueryKeyPrefix = (projectId: string) => [projectId, 'media-predictions'];

export const mediaPredictionsQueryOptions = ({
    projectId,
    selectedModel,
    mediaId,
    device,
    confidenceThreshold,
    range = null,
}: {
    projectId: string;
    selectedModel: SelectableModel | undefined;
    mediaId: string;
    device: string;
    confidenceThreshold: number | null;
    range?: PredictionVideoRangePayload | null;
}) =>
    queryOptions({
        queryKey: [
            ...getMediaPredictionsQueryKeyPrefix(projectId),
            mediaId,
            device,
            selectedModel?.modelId,
            selectedModel?.modelVariantId,
            confidenceThreshold,
            range,
        ],
        queryFn: async ({ signal }) => {
            if (selectedModel === undefined) return [];

            const response = await fetchClient.POST('/api/projects/{project_id}/dataset/media:predict', {
                signal,
                params: { path: { project_id: projectId } },
                body: {
                    ...getModelIdentifierPayload(selectedModel),
                    device,
                    confidence_threshold: confidenceThreshold,
                    media: [{ media_id: mediaId, range }],
                },
            });

            if (response.error) return [];

            const predictions = response.data?.predictions ?? [];

            return predictions.map((predictionItem) => {
                if ((predictionItem.prediction ?? []).length === 0) {
                    return {
                        ...predictionItem,
                        prediction: [
                            {
                                shape: { type: 'full_image' },
                                labels: [{ id: EMPTY_LABEL_ID }],
                                confidences: [1],
                            } satisfies PredictionDTO,
                        ],
                    };
                }

                return predictionItem;
            });
        },
        staleTime: 1000 * 60 * 5,
        enabled: selectedModel !== undefined,
    });

export const useMediaPredictions = ({
    mediaId,
    selectedModel,
    range,
    device,
    confidenceThreshold,
    enabled = true,
}: {
    mediaId: string;
    selectedModel: SelectableModel | undefined;
    range?: PredictionVideoRangePayload | null;
    device: string;
    confidenceThreshold: number | null;
    enabled?: boolean;
}) => {
    const projectId = useProjectIdentifier();
    const options = mediaPredictionsQueryOptions({
        projectId,
        selectedModel,
        mediaId,
        range,
        device,
        confidenceThreshold,
    });

    return useQuery({ ...options, enabled: options.enabled && enabled });
};

export const useIsFetchingMediaPredictions = ({
    mediaId,
    selectedModel,
    device,
    confidenceThreshold,
    range = null,
}: {
    mediaId: string;
    selectedModel: SelectableModel | undefined;
    device: string;
    confidenceThreshold: number | null;
    range?: PredictionVideoRangePayload | null;
}) => {
    const projectId = useProjectIdentifier();
    const { queryKey } = mediaPredictionsQueryOptions({
        projectId,
        selectedModel,
        mediaId,
        device,
        confidenceThreshold,
        range,
    });

    return useIsFetching({ queryKey, exact: true }) > 0;
};

export const useIsFetchingCurrentRangeFramesPredictions = (mediaId: string) => {
    const { selectedModel, selectedDevice, confidenceThreshold } = usePredictionSetup();
    const videoContext = useVideoPlayerContext();

    const frameNumber = videoContext?.videoFrame.frame_number ?? 0;
    const frameCount = videoContext?.videoFrame.frame_count ?? 1;

    // Exact query key for the range chunk covering the current frame (video only)
    const { startFrameIndex, endFrameIndex } = getVideoFrameRangeIndexes({
        frames: frameCount - 1,
        frameSkip: PREDICTION_FRAME_SKIP,
        frameNumber,
        chunkSize: PREDICTION_CHUNK_SIZE,
    });

    return useIsFetchingMediaPredictions({
        mediaId,
        selectedModel,
        device: selectedDevice,
        confidenceThreshold,
        range: { stride: PREDICTION_FRAME_SKIP, start_frame: startFrameIndex, end_frame: endFrameIndex },
    });
};

const useIsFetchingCurrentFramePredictions = (mediaId: string) => {
    const { selectedModel, selectedDevice, confidenceThreshold } = usePredictionSetup();
    const { mediaItem } = useSelectedMediaItem();

    const singleFrameRange = isVideoFrame(mediaItem)
        ? { start_frame: mediaItem.frame_number, end_frame: mediaItem.frame_number, stride: mediaItem.frame_stride }
        : null;

    return useIsFetchingMediaPredictions({
        mediaId,
        selectedModel,
        device: selectedDevice,
        confidenceThreshold,
        range: singleFrameRange,
    });
};

export const useIsFetchingPredictions = (mediaId: string) => {
    const videoContext = useVideoPlayerContext();

    const isPlaying = videoContext?.videoControls.isPlaying === true;

    const isFetchingRange = useIsFetchingCurrentRangeFramesPredictions(mediaId);
    const isFetchingSingleFrame = useIsFetchingCurrentFramePredictions(mediaId);

    return isPlaying ? isFetchingRange : isFetchingSingleFrame;
};
