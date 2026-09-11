// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision } from '@/api/types';
import { useTranslation } from '@/i18n';
import { dimensionValue, Flex, Heading, View } from '@geti-ui/ui';
import { useCancelJob, useGetCurrentRunningJobs } from 'hooks/api/jobs/jobs.hook';
import { isJobFailed } from 'hooks/api/util';
import { useDismissedJobs } from 'hooks/storage/use-dismissed-jobs.hook';
import { isEmpty, isNil } from 'lodash-es';

import { useGetTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { GroupByMode } from '../types';
import { FailedJobRow } from './failed-job-row.component';
import { RunningJobRow } from './running-job-row.component';
import { RunningJobTableHeader } from './running-job-table-header.component';

type CurrentRunningJobsProps = {
    groupBy: GroupByMode;
    datasetRevisions: DatasetRevision[];
};

export const CurrentRunningJobs = ({ groupBy, datasetRevisions }: CurrentRunningJobsProps) => {
    const { t } = useTranslation();
    const cancelJobMutation = useCancelJob();
    const activeRunningJobs = useGetCurrentRunningJobs();
    const { modelArchitectures } = useGetTaskModelArchitectures();
    const { dismissJob } = useDismissedJobs();

    const handleCancelRunning = (jobId: string | undefined) => {
        if (jobId) {
            cancelJobMutation.mutate({ params: { path: { job_id: jobId } } });
        }
    };

    if (isNil(activeRunningJobs) || isEmpty(activeRunningJobs)) {
        return null;
    }

    return (
        <Flex
            width={'100%'}
            gap={'size-200'}
            direction={'column'}
            UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-300)' }}
        >
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                {t('models.jobs.heading')}
            </Heading>

            <View backgroundColor={'gray-75'}>
                <RunningJobTableHeader groupBy={groupBy} />

                <div aria-label={'Current jobs'}>
                    {activeRunningJobs.map((job) =>
                        isJobFailed(job) ? (
                            <FailedJobRow
                                key={job.job_id}
                                job={job}
                                onDismiss={() => dismissJob(job.job_id)}
                                groupBy={groupBy}
                                datasetRevisions={datasetRevisions}
                                modelArchitectures={modelArchitectures}
                            />
                        ) : (
                            <RunningJobRow
                                key={job.job_id}
                                job={job}
                                onCancel={() => handleCancelRunning(job.job_id)}
                                groupBy={groupBy}
                                datasetRevisions={datasetRevisions}
                                modelArchitectures={modelArchitectures}
                            />
                        )
                    )}
                </div>
            </View>
        </Flex>
    );
};
