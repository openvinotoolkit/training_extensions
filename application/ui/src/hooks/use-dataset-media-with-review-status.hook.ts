// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useDatasetMediaFilterOptions } from './use-dataset-media-filter-options.hook';
import { useGetDatasetItemsById } from './use-get-dataset-items-by-id.hook';
import { useGetDatasetMediaItems } from './use-get-dataset-media-items.hook';

export const useDatasetMediaWithReviewStatus = () => {
    const filterOptions = useDatasetMediaFilterOptions();

    const mediaItemsResponse = useGetDatasetMediaItems(filterOptions);

    const datasetItemsResponse = useGetDatasetItemsById(filterOptions);

    const fetchNextPage = () => {
        if (mediaItemsResponse.hasNextPage && !mediaItemsResponse.isFetchingNextPage) {
            mediaItemsResponse.fetchNextPage();
        }

        if (datasetItemsResponse.hasNextPage && !datasetItemsResponse.isFetchingNextPage) {
            datasetItemsResponse.fetchNextPage();
        }
    };

    const isMediaItemReviewedById = (mediaItemId: string) => {
        return datasetItemsResponse.reviewStatus.get(mediaItemId) ?? false;
    };

    return {
        items: mediaItemsResponse.items,
        datasetItems: datasetItemsResponse.items,
        // Wait for both the media-items and review-status queries to settle
        // before declaring "ready". Otherwise the gallery flashes thumbnails
        // first and pops in the annotation-status badges a moment later, which
        // is a worse UX than a single brief loader.
        isPending: mediaItemsResponse.isPending || datasetItemsResponse.isPending,
        // Only true for actual next-page fetches on either query — never for
        // initial loads — so pagination doesn't trigger the full overlay.
        isFetchingNextPage: mediaItemsResponse.isFetchingNextPage || datasetItemsResponse.isFetchingNextPage,
        hasNextPage: mediaItemsResponse.hasNextPage || datasetItemsResponse.hasNextPage,
        totalCount: mediaItemsResponse.totalCount,
        fetchNextPage,
        isMediaItemReviewedById,
    };
};
