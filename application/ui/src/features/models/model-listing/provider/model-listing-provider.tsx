// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import type { DatasetRevision } from '@/api/types';
import { useGetDatasetRevisions } from 'hooks/use-get-dataset-revisions.hook';

import { useGetModels } from '../../hooks/api/use-get-models.hook';
import { useGroupedModels } from '../hooks/use-grouped-models.hook';
import type { GroupByMode, GroupedModels, SortBy, SortDescriptor } from '../types';
import { DEFAULT_SORT, DEFAULT_SORT_DIRECTIONS } from '../utils/sorting';

interface ModelListingContextValue {
    // State
    groupBy: GroupByMode;
    // Sort descriptor per group id; groups without an entry use the default sort.
    sortBy: Record<string, SortDescriptor>;
    expandedModelIds: Set<string>;
    groupedModels: GroupedModels[];
    searchBy: string;
    datasetRevisions: DatasetRevision[];
    showFailedModels: boolean;

    // Actions
    onGroupByChange: (mode: GroupByMode) => void;
    onSortChange: (groupId: string, key: SortBy) => void;
    onExpandModel: (modelId: string) => void;
    onSearchChange: (query: string) => void;
    onToggleShowFailedModels: () => void;
}

const ModelListingContext = createContext<ModelListingContextValue | null>(null);

interface ModelListingProviderProps {
    children: ReactNode;
}

export const ModelListingProvider = ({ children }: ModelListingProviderProps) => {
    const [groupBy, setGroupBy] = useState<GroupByMode>('dataset');
    const [sortBy, setSortBy] = useState<Record<string, SortDescriptor>>({});
    const [showFailedModels, setShowFailedModels] = useState<boolean>(true);
    const [expandedModelIds, setExpandedModelIds] = useState<Set<string>>(new Set());
    const [searchBy, setSearchBy] = useState<string>('');

    const { data: models } = useGetModels();
    const { data: datasetRevisions = [] } = useGetDatasetRevisions();
    const groupedModels = useGroupedModels(models, {
        groupBy,
        sortBy,
        searchBy,
        datasetRevisions,
        showFailedModels,
    });

    const onGroupByChange = (mode: GroupByMode) => {
        // Groups are keyed by dataset revision or by architecture, so their sort state does not carry over.
        setSortBy({});
        setGroupBy(mode);
    };

    const onSortChange = (groupId: string, key: SortBy) => {
        setSortBy((previous) => {
            const current = previous[groupId] ?? DEFAULT_SORT;

            return {
                ...previous,
                [groupId]:
                    current.key === key
                        ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
                        : { key, direction: DEFAULT_SORT_DIRECTIONS[key] },
            };
        });
    };

    const toggleShowFailedModels = () => {
        setShowFailedModels((prev) => !prev);
    };

    const onSearchChange = (query: string) => {
        setSearchBy(query);
    };

    const onExpandModel = (modelId: string) => {
        setExpandedModelIds((prev) => {
            const newExpandedModelIds = new Set(prev);

            if (newExpandedModelIds.has(modelId)) {
                newExpandedModelIds.delete(modelId);
            } else {
                newExpandedModelIds.add(modelId);
            }

            return newExpandedModelIds;
        });
    };

    const value: ModelListingContextValue = {
        groupBy,
        sortBy,
        expandedModelIds,
        groupedModels,
        searchBy,
        datasetRevisions,
        showFailedModels,

        onGroupByChange,
        onSortChange,
        onExpandModel,
        onSearchChange,
        onToggleShowFailedModels: toggleShowFailedModels,
    };

    return <ModelListingContext.Provider value={value}>{children}</ModelListingContext.Provider>;
};

export const useModelListing = (): ModelListingContextValue => {
    const context = useContext(ModelListingContext);

    if (!context) {
        throw new Error('useModelListing must be used within a ModelListingProvider');
    }

    return context;
};
