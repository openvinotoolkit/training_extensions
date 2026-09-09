// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, PointerEvent, ReactNode, RefObject, useMemo, useRef } from 'react';

import type { Media } from '@/api/types';
import { ZoomTransform } from '@/components/zoom/zoom-transform';
import { Loading } from '@geti-ui/ui';
import { useIsFetching } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useSpinDelay } from 'spin-delay';

import { loadImageQueryOptions } from '../hooks/use-load-image-query.hook';
import { MediaImage } from './media-image.component';

type MediaCanvasProps = {
    mediaItem: Media;
    image: ImageData;
    containerRef?: RefObject<HTMLDivElement | null>;
    onPointerMove?: (event: PointerEvent<HTMLDivElement>) => void;
    className?: string;
    isLoadingOverlay?: boolean;
    children?: ReactNode;
};

export const MediaCanvas = ({
    mediaItem,
    image,
    containerRef,
    onPointerMove,
    className,
    isLoadingOverlay = false,
    children,
}: MediaCanvasProps) => {
    const projectId = useProjectIdentifier();
    const localRef = useRef<HTMLDivElement | null>(null);
    const resolvedRef = containerRef ?? localRef;

    const isFetchingMedia = useIsFetching({ queryKey: loadImageQueryOptions(projectId, mediaItem).queryKey }) > 0;
    const isLoadingMedia = useSpinDelay(isFetchingMedia, { delay: 400, minDuration: 200 });

    // A new object would make the zoom refit on every render, discarding the user's pan and zoom
    const size = useMemo(
        () => ({ width: mediaItem.width, height: mediaItem.height }),
        [mediaItem.width, mediaItem.height]
    );

    return (
        <div style={{ position: 'relative', height: '100%', width: '100%' }}>
            <ZoomTransform target={size}>
                <div
                    ref={resolvedRef}
                    style={{ position: 'relative', height: '100%', width: '100%' }}
                    onContextMenu={(event: MouseEvent): void => event.preventDefault()}
                    onPointerMove={onPointerMove}
                    className={className}
                >
                    <MediaImage image={image} mediaItem={mediaItem} />
                    {children}
                </div>
            </ZoomTransform>

            {/* Rendered outside the zoom transform so it is not scaled along with the media */}
            {(isLoadingMedia || isLoadingOverlay) && (
                <Loading
                    mode={'overlay'}
                    style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
                    aria-label={'Media canvas loading'}
                />
            )}
        </div>
    );
};
