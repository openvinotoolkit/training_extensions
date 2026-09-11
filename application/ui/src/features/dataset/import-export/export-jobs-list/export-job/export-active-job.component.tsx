// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ExportDatasetJob } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Divider, Flex, Loading, Text, View } from '@geti-ui/ui';
import { getJobProgress, isJobRunning } from 'hooks/api/util';
import { useExportDataset } from 'hooks/storage/use-export-dataset.hook';

import { BottomProgressBar } from '../../../../models/model-listing/current-running-jobs/bottom-progress-bar.component';
import { CancelJobConfirmation } from '../../cancel-job-confirmation/cancel-job-confirmation.component';
import { ExportJobDetails } from './export-details/export-details.component';

type ExportActiveJobProps = {
    job: ExportDatasetJob;
    datasetName?: string;
};

export const ExportActiveJob = ({ job, datasetName }: ExportActiveJobProps) => {
    const { t } = useTranslation();
    const isRunning = isJobRunning(job);
    const { removeLsExportId } = useExportDataset();

    const progress = getJobProgress(job?.progress);

    const handleRemove = () => {
        removeLsExportId(job.job_id);
    };

    return (
        <BottomProgressBar progress={progress}>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <ExportJobDetails metadata={job.metadata} datasetName={datasetName} />
                    <CancelJobConfirmation jobId={job.job_id} onRemove={handleRemove} />
                </Flex>

                <Text>{t('dataset.export.processing')}</Text>

                <Divider size='S' marginY='size-150' />

                <Flex justifyContent='space-between'>
                    <Flex alignItems='center' gap='size-100'>
                        <Loading mode='inline' size='S' />
                        <Text UNSAFE_style={{ textTransform: 'capitalize' }}>
                            {job?.message ?? job.status.toLocaleLowerCase()}
                        </Text>
                    </Flex>

                    {isRunning && <Text>{progress}%</Text>}
                </Flex>
            </View>
        </BottomProgressBar>
    );
};
