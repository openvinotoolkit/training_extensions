// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media, MediaWithPagination } from '@/api/types';
import { InfiniteData, QueryClient } from '@tanstack/react-query';

import { isVideo } from '../../../../shared/media-item-utils';

export const incrementCachedAnnotatedFrameCount = (queryClient: QueryClient, mediaItem: Media) => {
    queryClient.setQueriesData<InfiniteData<MediaWithPagination>>(
        { queryKey: ['get', '/api/projects/{project_id}/dataset/media'] },
        (oldData) => {
            if (!oldData?.pages) return oldData;

            return {
                ...oldData,
                pages: oldData.pages.map((page) => ({
                    ...page,
                    items: page.items.map((item) =>
                        isVideo(item) && item.id === mediaItem.id
                            ? { ...item, annotated_frame_count: item.annotated_frame_count + 1 }
                            : item
                    ),
                })),
            };
        }
    );
};
