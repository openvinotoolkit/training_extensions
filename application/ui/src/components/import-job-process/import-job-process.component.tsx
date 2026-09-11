// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { dimensionValue, Flex, Loading, Text } from '@geti-ui/ui';
import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { getJobProgress, isJobPending } from 'hooks/api/util';
import { isEmpty } from 'lodash-es';

import { CircularProgress } from '../circular-progress/circular-progress.component';
import { ThreeDotsFlashing } from '../three-dots-flashing/three-dots-flashing.component';

import classes from './import-job-process.module.scss';

type ImportJobProcessProps = {
    jobId: string | null | undefined;
    fileName: string;
    message: string;
    onError: () => void;
    onSuccess: () => void;
};

export const ImportJobProcess = ({ jobId, fileName, message, onError, onSuccess }: ImportJobProcessProps) => {
    const { t } = useTranslation();
    const { data: job, isFetching, isPending } = useImportJobStatus({ jobId, onError, onSuccess });

    const progress = getJobProgress(job?.progress);
    const isPreparingJobLoading = isJobPending(job) && isPending;

    if (!isFetching && isEmpty(job)) {
        return null;
    }

    return (
        <Flex
            width='100%'
            height='100%'
            gap='size-275'
            direction='column'
            alignItems='center'
            justifyContent='center'
            UNSAFE_style={{ padding: dimensionValue('size-500') }}
        >
            {!isPreparingJobLoading && (
                <CircularProgress
                    size={80}
                    percentage={progress}
                    strokeWidth={8}
                    labelFontSize={12}
                    color='static-blue-200'
                    labelFontColor='gray-700'
                    backStrokeColor='gray-75'
                />
            )}

            {isPreparingJobLoading && <Loading mode='inline' size='L' style={{ height: 'auto' }} />}

            <Flex direction='column' alignItems='center' justifyContent='center'>
                <Text UNSAFE_className={classes.title}>
                    {t('dataset.import.preparingJob')}
                    <ThreeDotsFlashing />
                </Text>
                <Text UNSAFE_className={classes.description}>{message}</Text>

                <Text marginTop='size-100'>{fileName}</Text>
            </Flex>
        </Flex>
    );
};
