// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef } from 'react';

import type { AnnotationDTO, Media } from '@/api/types';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { range } from 'lodash-es';
import { useLocalStorage } from 'usehooks-ts';

import { isVideoFrame } from '../../../shared/media-item-utils';
import type { AnnotatorMode } from '../annotator-mode';
import { usePrefetchMediaItem } from '../hooks/use-prefetch-media-item.hook';
import { useVideoPlayerContext } from '../video-player/video-player-provider.component';

export const getInitialAnnotations = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? annotationsDTO : [];
};

export const getNextMediaItem = (currentMediaItem: Media, allMediaItems: Media[], step: number): Media | undefined => {
    if (isVideoFrame(currentMediaItem)) {
        const videoFrames = range(0, currentMediaItem.frame_count, step);
        const currentIndex = videoFrames.findIndex((frame) => frame === currentMediaItem.frame_number);

        if (currentIndex >= 0 && currentIndex < videoFrames.length - 1) {
            return {
                ...currentMediaItem,
                frame_number: videoFrames[currentIndex + 1],
            };
        } else {
            const nextFrame = videoFrames.find((frame) => frame > currentMediaItem.frame_number);

            if (nextFrame !== undefined) {
                return {
                    ...currentMediaItem,
                    frame_number: nextFrame ?? 0,
                };
            }
        }
    }

    const currentIndex = allMediaItems.findIndex(({ id }) => id === currentMediaItem.id);

    if (currentIndex < 0) {
        return allMediaItems[0];
    }

    if (currentIndex >= allMediaItems.length - 1) {
        return undefined;
    }

    return allMediaItems[currentIndex + 1];
};

export const useNextMediaItem = (currentMediaItem: Media, allMediaItems: Media[]) => {
    const context = useVideoPlayerContext();
    const step = context?.step ?? 1;

    return useMemo(
        () => getNextMediaItem(currentMediaItem, allMediaItems, step),
        [allMediaItems, currentMediaItem, step]
    );
};

// Prefetches the next media item so sequential navigation hits the React Query cache.
export const useNextMediaPrefetch = (currentMediaItem: Media, allMediaItems: Media[]) => {
    const nextMediaItem = useNextMediaItem(currentMediaItem, allMediaItems);

    usePrefetchMediaItem(nextMediaItem);

    return { nextMediaItem };
};

export const useAnnotatorMode = () => {
    const projectId = useProjectIdentifier();

    const [mode, setMode] = useLocalStorage<AnnotatorMode>(`${projectId}-annotator-mode`, 'annotation');

    return [mode, setMode] as const;
};

export const usePlayPauseVideoBySystem = (isLoadingRangePredictions: boolean) => {
    const isPausedBySystem = useRef<boolean>(false);
    const context = useVideoPlayerContext();

    const playRef = useRef(context?.videoControls.play);
    const pauseRef = useRef(context?.videoControls.pause);

    useEffect(() => {
        playRef.current = context?.videoControls.play;
    }, [context?.videoControls.play]);

    useEffect(() => {
        pauseRef.current = context?.videoControls.pause;
    }, [context?.videoControls.pause]);

    useEffect(() => {
        if (isLoadingRangePredictions && context?.videoControls.isPlaying) {
            isPausedBySystem.current = true;
            pauseRef.current?.();
        } else if (!isLoadingRangePredictions && isPausedBySystem.current) {
            isPausedBySystem.current = false;
            playRef.current?.();
        }
    }, [isLoadingRangePredictions, context?.videoControls.isPlaying]);
};
