// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { DatasetStatistics } from '@/components/dataset-statistics/dataset-statistics.component';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

type DatasetRevisionStatisticsProps = {
    datasetRevisionId: string;
};

export const DatasetRevisionStatistics = ({ datasetRevisionId }: DatasetRevisionStatisticsProps) => {
    const projectId = useProjectIdentifier();

    const { data: annotatedItems } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items',
        {
            params: {
                path: { project_id: projectId, dataset_revision_id: datasetRevisionId },
                query: { limit: 1, annotation_status: 'reviewed' },
            },
        }
    );

    const { data: mediaItems } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items',
        {
            params: { path: { project_id: projectId, dataset_revision_id: datasetRevisionId } },
        }
    );

    const totalMediaItems = mediaItems?.pagination.total ?? 0;
    const totalAnnotatedItems = annotatedItems?.pagination.total ?? 0;

    return (
        <DatasetStatistics label='items' totalMediaItems={totalMediaItems} totalAnnotatedItems={totalAnnotatedItems} />
    );
};
