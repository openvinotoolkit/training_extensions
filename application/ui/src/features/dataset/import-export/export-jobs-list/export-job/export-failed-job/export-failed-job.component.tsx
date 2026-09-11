// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ExportDatasetJob } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Button, Divider, Flex, Text, View } from '@geti-ui/ui';

import { useExportDataset } from '../../../../../../hooks/storage/use-export-dataset.hook';
import { ExportJobDetails } from '../export-details/export-details.component';

type ExportFailedJobProps = {
    job: ExportDatasetJob;
    datasetName?: string;
};

export const ExportFailedJob = ({ job, datasetName }: ExportFailedJobProps) => {
    const { t } = useTranslation();
    const { removeLsExportId } = useExportDataset();

    const handleClose = () => {
        removeLsExportId(job.job_id);
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <ExportJobDetails metadata={job.metadata} datasetName={datasetName} />

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close export dataset status'
                        onPress={handleClose}
                    >
                        {t('dataset.export.close')}
                    </Button>
                </Flex>
            </Flex>

            <Text>{job.message}</Text>
            <Divider size='S' marginY='size-150' />
            <Text>{t('dataset.export.error', { error: job.error })}</Text>
        </View>
    );
};
