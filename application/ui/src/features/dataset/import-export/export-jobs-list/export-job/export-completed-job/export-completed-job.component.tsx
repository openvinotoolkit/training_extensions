// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '@/api';
import type { ExportDatasetJob } from '@/api/types';
import { Button, Divider, Flex, Text, View } from '@geti-ui/ui';
import { useDeleteStagedDataset, useStagedDataset } from 'hooks/api/staged-dataset.hook';
import { isNil } from 'lodash-es';

import { useExportDataset } from '../../../../../../hooks/storage/use-export-dataset.hook';
import { downloadFile } from '../../../../../../platform/download-file';
import { ExportJobDetails } from '../export-details/export-details.component';

type ExportCompletedJobProps = {
    job: ExportDatasetJob;
    datasetName?: string;
};

export const ExportCompletedJob = ({ job, datasetName }: ExportCompletedJobProps) => {
    const { removeLsExportId } = useExportDataset();
    const stageDatasetResponse = useStagedDataset(job.metadata.dataset_id);

    const removeStagedDatasetMutation = useDeleteStagedDataset({
        stagedDatasetId: job.metadata.dataset_id,
        deleteEntry: () => removeLsExportId(job.job_id),
    });

    const hasInvalidStagedDataset = isNil(job.metadata.dataset_id);
    const message = hasInvalidStagedDataset ? job.message : 'Dataset is ready for download';

    const handleClose = () => {
        if (hasInvalidStagedDataset) {
            removeLsExportId(job.job_id);
        } else {
            removeStagedDatasetMutation.mutate();
        }
    };

    const handleDownload = () => {
        const url = `${API_BASE_URL}/api/staged_datasets/${job.metadata.dataset_id}/zip`;

        downloadFile(url, `dataset_${job.metadata.dataset_id}.zip`, 'Dataset download started');
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='end' gap='size-250' marginBottom='size-125'>
                <ExportJobDetails metadata={job.metadata} datasetName={datasetName} />

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close export dataset status'
                        onPress={handleClose}
                        isPending={removeStagedDatasetMutation.isPending}
                        isDisabled={removeStagedDatasetMutation.isPending}
                    >
                        Close
                    </Button>
                    <Button
                        variant='secondary'
                        aria-label='download dataset'
                        onPress={handleDownload}
                        isPending={stageDatasetResponse.isFetching}
                        isDisabled={
                            hasInvalidStagedDataset ||
                            stageDatasetResponse.isFetching ||
                            removeStagedDatasetMutation.isPending
                        }
                    >
                        Download
                    </Button>
                </Flex>
            </Flex>

            <Divider size={'S'} marginY={'size-150'} />

            <Text>{message}</Text>
        </View>
    );
};
