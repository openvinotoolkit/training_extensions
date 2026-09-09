// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useRef } from 'react';

import type { DatasetItemAnnotationStatus, DatasetSubset } from '@/api/types';

import { type SortDirection } from './sort-direction.interface';
import { useGetDatasetItems } from './use-get-dataset-items.hook';

type UseGetDatasetItemsByIdOptions = {
    annotationStatus?: DatasetItemAnnotationStatus;
    sortDirection?: SortDirection;
    subsets?: DatasetSubset[];
    labelIds?: string[];
    startDate?: string;
    endDate?: string;
    datasetViewId?: string;
};

export const useGetDatasetItemsById = ({
    annotationStatus,
    sortDirection,
    subsets,
    labelIds,
    startDate,
    endDate,
    datasetViewId,
}: UseGetDatasetItemsByIdOptions) => {
    const datasetItemsQuery = useGetDatasetItems({
        annotationStatus,
        sortDirection,
        subsets,
        labelIds,
        startDate,
        endDate,
        datasetViewId,
    });

    const accumulatedReviewStatusRef = useRef(new Map<string, boolean>());

    const reviewStatus = useMemo(() => {
        datasetItemsQuery.items.forEach(({ id, user_reviewed }) => {
            accumulatedReviewStatusRef.current.set(id, user_reviewed);
        });

        return new Map(accumulatedReviewStatusRef.current);
    }, [datasetItemsQuery.items]);

    return { reviewStatus, ...datasetItemsQuery };
};
