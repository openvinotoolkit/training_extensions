// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import type { Media, MediaVideo } from '@/api/types';
import { Flex, Skeleton } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { useIsScrolling } from '../../hooks/use-is-scrolling.hook';
import { isVideo } from '../../shared/media-item-utils';
import { formatCompactDuration } from './util';

import classes from './media-thumbnail.module.scss';

type MediaThumbnailProps = {
    onClick?: () => void;
    onDoubleClick?: () => void;
    url: string;
    alt: string;
    item: Pick<Media, 'type'> | Pick<MediaVideo, 'type' | 'frame_count' | 'annotated_frame_count' | 'duration'>;
};

type VideoIndicatorProps = {
    duration: number;
};

const VideoIndicator = ({ duration }: VideoIndicatorProps) => {
    return (
        <Flex
            gap={'size-50'}
            left={'size-50'}
            bottom={'size-50'}
            position={'absolute'}
            alignItems={'center'}
            UNSAFE_className={classes.videoIndicator}
        >
            {formatCompactDuration(duration)}
        </Flex>
    );
};

export const MediaThumbnail = ({ onDoubleClick, onClick, url, alt, item }: MediaThumbnailProps) => {
    const imgRef = useRef<HTMLImageElement>(null);
    const isScrolling = useIsScrolling();
    const [status, setStatus] = useState<'loading' | 'loaded' | 'error'>('loading');

    // Tiles that fly past during a fast scroll would otherwise start a request and immediately
    // cancel it; enough of those wedge the connection and leave the gallery blank.
    useEffect(() => {
        if (isScrolling || imgRef.current === null) {
            return;
        }

        // Re-assigning the same value restarts the fetch, so every scroll stop would abort and
        // re-request the thumbnails that are already loading.
        if (imgRef.current.getAttribute('src') === url) {
            return;
        }

        setStatus('loading');
        imgRef.current.src = url;
    }, [url, isScrolling]);

    useEffect(() => {
        const ref = imgRef.current;

        return () => {
            if (ref) {
                ref.src = '';
            }
        };
    }, []);

    return (
        <div onDoubleClick={onDoubleClick} onClick={onClick} className={classes.imgContainer}>
            <img
                ref={imgRef}
                alt={alt}
                className={clsx(classes.img, {
                    [classes.imgHidden]: status !== 'loaded',
                })}
                draggable={false}
                decoding={'async'}
                onLoad={() => setStatus('loaded')}
                onError={() => {
                    if (imgRef.current?.complete === true) {
                        setStatus('error');
                    }
                }}
            />
            {status === 'loading' && <Skeleton width={'100%'} height={'100%'} className={classes.skeleton} />}
            {isVideo(item) && <VideoIndicator duration={item.duration} />}
        </div>
    );
};
