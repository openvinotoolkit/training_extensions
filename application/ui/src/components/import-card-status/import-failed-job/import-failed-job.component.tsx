// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Text } from '@geti-ui/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes, isNonEmptyString } from '../../../shared/util';
import { JobStatusCard } from '../../job-status-card/job-status-card.component';

import classes from './import-failed-job.module.scss';

type ImportFailedJobProps = {
    size: number;
    error?: string;
    message?: string;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
};

const TechnicalDetails = ({ error }: { error: string }) => {
    const { t } = useTranslation();

    return (
        <details className={classes.details} aria-label={'Technical details of the job failure'}>
            <summary className={classes.summary}>{t('dataset.import.technicalDetails')}</summary>
            <pre className={classes.traceback}>{error}</pre>
        </details>
    );
};

const BottomMessage = ({ error, message }: { error: string; message: string }) => {
    return (
        <>
            <Text>{message}</Text>
            <TechnicalDetails error={error} />
        </>
    );
};

export const ImportFailedJob = ({
    size,
    error,
    message,
    fileName,
    stagedDatasetId,
    deleteEntry,
}: ImportFailedJobProps) => {
    const { t } = useTranslation();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const errorMessage = isNonEmptyString(message) ? message : t('dataset.import.unknownError');
    const errorDetails = isNonEmptyString(error) ? error : undefined;

    return (
        <JobStatusCard
            title={t('dataset.import.jobTitle', { fileName, size: formatBytes(size) })}
            actionButtons={
                <Button
                    variant='secondary'
                    style='fill'
                    aria-label='close import dataset status'
                    onPress={deleteFileMutation.mutate}
                    isPending={deleteFileMutation.isPending}
                    isDisabled={deleteFileMutation.isPending}
                >
                    {t('dataset.import.close')}
                </Button>
            }
            bottomLeftMessage={
                errorDetails ? <BottomMessage error={errorDetails} message={errorMessage} /> : errorMessage
            }
        />
    );
};
