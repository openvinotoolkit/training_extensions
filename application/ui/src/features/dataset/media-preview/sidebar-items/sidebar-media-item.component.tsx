// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';
import { MediaItem } from '@/components/media-item/media-item.component';
import { MediaThumbnail } from '@/components/media-thumbnail/media-thumbnail.component';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { getThumbnailUrl } from '../../../../shared/media-url.utils';
import { AnnotationStatusIcon } from '../../gallery/annotation-state-icon.component';

type SidebarMediaItemProps = {
    item: Media;
    isUserReviewed: boolean;
    onSelectedMediaItem: (item: Media) => void;
};

export const SidebarMediaItem = ({ item, isUserReviewed, onSelectedMediaItem }: SidebarMediaItemProps) => {
    const projectId = useProjectIdentifier();

    return (
        <MediaItem
            contentElement={() => (
                <MediaThumbnail
                    item={item}
                    alt={item.name}
                    url={getThumbnailUrl(projectId, String(item.id))}
                    onClick={() => onSelectedMediaItem(item)}
                />
            )}
            bottomRightElement={() => {
                return <AnnotationStatusIcon isReviewed={isUserReviewed} />;
            }}
        />
    );
};
