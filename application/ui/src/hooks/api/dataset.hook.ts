// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { useQuery, useSuspenseQuery } from '@tanstack/react-query';
import { useDatasetViewId } from 'hooks/use-dataset-view-id.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

const datasetStatisticsQueryOptions = (projectId: string, datasetViewId: string | null) =>
    $api.queryOptions('get', '/api/projects/{project_id}/dataset/statistics', {
        params: {
            path: { project_id: projectId },
            query: {
                dataset_view_id: datasetViewId,
            },
        },
    });

export const useDatasetStatistics = () => {
    const projectId = useProjectIdentifier();
    const [datasetViewId] = useDatasetViewId();

    return useSuspenseQuery(datasetStatisticsQueryOptions(projectId, datasetViewId));
};

// Non suspending variant, for places where the statistics are optional and should not block rendering
export const useDatasetStatisticsQuery = (enabled = true) => {
    const projectId = useProjectIdentifier();
    const [datasetViewId] = useDatasetViewId();

    return useQuery({ ...datasetStatisticsQueryOptions(projectId, datasetViewId), enabled });
};
