// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';
import { usePrefetchQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { isVideoFrame } from '../../../shared/media-item-utils';
import { mediaPredictionsQueryOptions } from '../../annotator/api/use-media-predictions';
import { usePredictionSetup } from '../../annotator/predictions-setup-provider.component';

export const useNextPredictionPrefetch = (nextMediaItem: Media) => {
    const projectId = useProjectIdentifier();
    const { selectedModel, selectedDevice, confidenceThreshold } = usePredictionSetup();

    const range = isVideoFrame(nextMediaItem)
        ? {
              stride: nextMediaItem.frame_stride,
              end_frame: nextMediaItem.frame_number,
              start_frame: nextMediaItem.frame_number,
          }
        : null;

    usePrefetchQuery(
        mediaPredictionsQueryOptions({
            projectId,
            selectedModel,
            mediaId: nextMediaItem.id,
            device: selectedDevice,
            confidenceThreshold,
            range,
        })
    );
};
