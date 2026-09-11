// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media, MediaVideo } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Content, ContextualHelp, Divider, Text } from '@geti-ui/ui';

import { isVideo } from '../../../../shared/media-item-utils';

import classes from './media-item-contextual-help.module.scss';

type MediaItemContextualHelpProps = {
    item: Pick<Media, 'type'> | Pick<MediaVideo, 'type' | 'frame_count' | 'annotated_frame_count' | 'duration'>;
};

export const MediaItemContextualHelp = ({ item }: MediaItemContextualHelpProps) => {
    const { t } = useTranslation();

    if (!isVideo(item)) {
        return null;
    }

    return (
        <>
            <ContextualHelp
                variant='info'
                UNSAFE_className={classes.videoIndicatorDetails}
                aria-label='Media information'
            >
                <Content>
                    <Text>{t('dataset.gallery.mediaInfo.annotatedFrames', { count: item.annotated_frame_count })}</Text>
                    <br />
                    <Text>{t('dataset.gallery.mediaInfo.totalFrames', { count: item.frame_count })}</Text>
                </Content>
            </ContextualHelp>

            <Divider orientation={'vertical'} size={'S'} />
        </>
    );
};
