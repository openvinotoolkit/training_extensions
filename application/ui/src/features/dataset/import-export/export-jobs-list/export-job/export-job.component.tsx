// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { View } from '@geti-ui/ui';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isNil, isString } from 'lodash-es';

import { useExportStatus } from '../hooks/use-export-status.hook';
import { ExportActiveJob } from './export-active-job.component';
import { ExportCompletedJob } from './export-completed-job/export-completed-job.component';
import { ExportFailedJob } from './export-failed-job/export-failed-job.component';

type ExportJobProps = {
    jobId: string;
    datasetId: string | null;
};

export const ExportJob = ({ jobId, datasetId }: ExportJobProps) => {
    const projectId = useProjectIdentifier();
    const { data: job } = useExportStatus(jobId);

    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    const { data: datasetDetails } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}',
        { params: { path: { project_id: projectId, dataset_revision_id: datasetId ?? '' } } },
        { enabled: isString(datasetId) }
    );

    if (isNil(job)) {
        return null;
    }

    return (
        <View
            position='relative'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
            borderWidth='thin'
        >
            {isJobDone(job) && <ExportCompletedJob job={job} datasetName={datasetDetails?.name} />}
            {isJobFailed(job) && <ExportFailedJob job={job} datasetName={datasetDetails?.name} />}
            {isRunningOrPending && <ExportActiveJob job={job} datasetName={datasetDetails?.name} />}
        </View>
    );
};
