// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useDatasetViewId } from 'hooks/use-dataset-view-id.hook';
import { isEmpty } from 'lodash-es';

import { useDatasetFiltersSearchParams } from './use-dataset-filters-search-params.hook';
import type { UseGetDatasetMediaItemsOptions } from './use-get-dataset-media-items.hook';

/**
 * Resolves the dataset filters currently applied through the URL into the options
 * shape accepted by the media/dataset item queries.
 */
export const useDatasetMediaFilterOptions = (): UseGetDatasetMediaItemsOptions => {
    const { selectedLabelIds, annotationStatus, startDate, endDate, sortDirection, selectedSubsets } =
        useDatasetFiltersSearchParams();
    const [datasetViewId] = useDatasetViewId();

    return {
        annotationStatus: annotationStatus ?? undefined,
        labelIds: isEmpty(selectedLabelIds) ? undefined : selectedLabelIds,
        startDate: startDate ?? undefined,
        endDate: endDate ?? undefined,
        datasetViewId: datasetViewId ?? undefined,
        subsets: isEmpty(selectedSubsets) ? undefined : selectedSubsets,
        sortDirection,
    };
};
