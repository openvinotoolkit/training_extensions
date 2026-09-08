// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { $api } from '@/api';
import type { DatasetItemAnnotationStatus, DatasetSubset, Media, MediaDTO, Pagination } from '@/api/types';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import isEmpty from 'lodash-es/isEmpty';

import { type SortDirection } from './sort-direction.interface';

const DATASET_ITEMS_LIMIT = 40;

type SortBy = 'upload_date';

export interface UseGetDatasetMediaItemsOptions {
    subsets?: DatasetSubset[];
    annotationStatus?: DatasetItemAnnotationStatus;
    labelIds?: string[];
    startDate?: string;
    endDate?: string;
    sortDirection?: SortDirection;
    datasetViewId?: string;
}

export interface DatasetMediaQueryFilters {
    subsets?: DatasetSubset[];
    labels?: string[];
    end_date?: string;
    start_date?: string;
    annotation_status?: DatasetItemAnnotationStatus;
    sort_direction?: SortDirection;
    sort_by?: SortBy;
    dataset_view_id?: string;
}

export const buildDatasetMediaQueryFilters = (options?: UseGetDatasetMediaItemsOptions): DatasetMediaQueryFilters => {
    const query: DatasetMediaQueryFilters = {};

    if (options !== undefined && !isEmpty(options?.subsets)) {
        query.subsets = options.subsets;
    }

    if (options?.annotationStatus !== undefined) {
        query.annotation_status = options.annotationStatus;
    }

    if (options?.labelIds !== undefined) {
        query.labels = options.labelIds;
    }

    if (options?.startDate !== undefined) {
        query.start_date = options.startDate;
    }

    if (options?.endDate !== undefined) {
        query.end_date = options.endDate;
    }

    if (options?.sortDirection !== undefined) {
        query.sort_direction = options.sortDirection;
        query.sort_by = 'upload_date';
    }

    if (options?.datasetViewId !== undefined) {
        query.dataset_view_id = options.datasetViewId;
    }

    return query;
};

const getMediaEntities = (items: MediaDTO[]): Media[] => {
    return items.map((item) => {
        // We will never get the video frame using '/api/projects/{project_id}/dataset/media', it's added only because
        // of documentation reasons. We use MediaVideoFrame as a local type to work with the played frame in the video.
        if (item.type === 'video_frame') {
            return {
                duration: 0,
                frame_count: 0,
                annotated_frame_count: 0,
                fps: 0,
                frame_number: 0,
                frame_stride: 0,
                ...item,
            };
        }

        return item;
    });
};

export const useGetDatasetMediaItems = (options?: UseGetDatasetMediaItemsOptions) => {
    const project_id = useProjectIdentifier();

    const query = {
        offset: 0,
        limit: DATASET_ITEMS_LIMIT,
        ...buildDatasetMediaQueryFilters(options),
    };

    const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isPending } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/dataset/media',
        {
            params: {
                query,
                path: { project_id },
            },
        },
        {
            pageParamName: 'offset',
            initialPageParam: 0,
            getNextPageParam: ({ pagination }: { pagination: Pagination }) => {
                const total = pagination.offset + pagination.count;

                if (total >= pagination.total) {
                    return undefined;
                }

                return pagination.offset + DATASET_ITEMS_LIMIT;
            },
        }
    );

    const items = useMemo(() => {
        const mediaItems = data?.pages?.flatMap((page) => page.items) ?? [];

        return getMediaEntities(mediaItems);
    }, [data?.pages]);

    const totalCount = data?.pages[0]?.pagination?.total ?? 0;

    return { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount };
};
