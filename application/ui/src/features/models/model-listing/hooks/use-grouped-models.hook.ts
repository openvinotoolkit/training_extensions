// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import type { DatasetRevision, Model } from '@/api/types';

import type { GroupByMode, GroupedModels, SortDescriptor } from '../types';
import {
    filterBySearch,
    filterOutFailedModels,
    filterOutTrainingModels,
    groupModels,
    removeEmpty,
    sortGroupedModels,
} from '../utils/model-transforms';
import { sortGroupedModelsByDatasetRevisionDate } from '../utils/sorting';

type UseGroupedModelsOptions = {
    groupBy: GroupByMode;
    // Each group sorts independently; groups without an entry use the default sort.
    sortBy: Record<string, SortDescriptor>;
    searchBy: string;
    datasetRevisions: DatasetRevision[];
    showFailedModels: boolean;
};

// Responsible for:
// - Filtering models based on searchBy query and failed status (only models that are not currently training)
// - Grouping models based on the selected grouping mode
// - Sorting models within each group based on the selected sorting criteria
export const useGroupedModels = (models: Model[] | undefined, options: UseGroupedModelsOptions): GroupedModels[] => {
    const { groupBy, sortBy, searchBy, datasetRevisions, showFailedModels } = options;

    return useMemo(() => {
        if (!models) return [];

        const filteredByTraining = filterOutTrainingModels(models);
        const filteredByFailedModels = showFailedModels
            ? filteredByTraining
            : filterOutFailedModels(filteredByTraining);
        const filteredBySearch = filterBySearch(filteredByFailedModels, searchBy);
        const grouped = groupModels(filteredBySearch, groupBy, datasetRevisions);
        const sortedModelsInsideGroup = sortGroupedModels(grouped, sortBy, datasetRevisions);
        const sortedGroupsByDatasetRevisionDate = sortGroupedModelsByDatasetRevisionDate(
            sortedModelsInsideGroup,
            datasetRevisions
        );

        return removeEmpty(sortedGroupsByDatasetRevisionDate);
    }, [models, groupBy, sortBy, searchBy, datasetRevisions, showFailedModels]);
};
