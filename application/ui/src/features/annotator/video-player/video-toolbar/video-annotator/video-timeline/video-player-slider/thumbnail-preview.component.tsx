// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { MediaVideoFrame } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Image, View } from '@geti-ui/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { getVideoFrameThumbnailUrl } from '../../../../../../../shared/media-url.utils';
import { formatDurationText } from '../../../time-utils';
import { FrameNumberIndicator } from './frame-number-indicator.component';

interface ThumbnailPreviewProps {
    frameNumber: number;
    videoFrame: MediaVideoFrame;
    width: number;
    height: number;
    x: number;
}

export const ThumbnailPreview = ({ videoFrame, frameNumber, width, height, x }: ThumbnailPreviewProps) => {
    // TODO: Use it when video frame navigation is ready and API supports it.
    // const constructVideoFrame = useConstructVideoFrame(mediaItem);
    // const videoFrame = constructVideoFrame(frameNumber) as VideoFrame;
    const { t } = useTranslation();
    const projectIdentifier = useProjectIdentifier();

    const fps = videoFrame.fps;
    const src = getVideoFrameThumbnailUrl(projectIdentifier, videoFrame.id, frameNumber);

    const durationText = formatDurationText(frameNumber / fps);

    return (
        <View
            position='absolute'
            top={-height - 25}
            left={x - width / 2}
            overflow='hidden'
            borderRadius='small'
            UNSAFE_style={{ boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.38)', pointerEvents: 'none' }}
        >
            <View position='fixed'>
                <FrameNumberIndicator frameNumber={frameNumber} />
                <Image
                    src={src}
                    alt={t('annotator.video.frames.thumbnailAlt', { frameNumber })}
                    objectFit='cover'
                    height={height}
                    width={width}
                />
                <View
                    backgroundColor='gray-50'
                    paddingX='size-75'
                    paddingY='size-25'
                    alignSelf='center'
                    UNSAFE_style={{ color: 'white', fontSize: '11px', textAlign: 'center' }}
                >
                    {durationText}
                </View>
                <View backgroundColor='static-white' width='1px' marginX='auto' height='size-200'>
                    &nbsp;
                </View>
            </View>
        </View>
    );
};
