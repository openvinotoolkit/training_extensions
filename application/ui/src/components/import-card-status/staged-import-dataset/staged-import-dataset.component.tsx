// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button } from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';
import { useStagedDataset } from 'hooks/api/staged-dataset.hook';

import { getErrorMessage } from '../../../query-client/query-client';
import { formatBytes } from '../../../shared/util';
import { DeleteStagedFileConfirmation } from '../../delete-staged-file-confirmation/delete-staged-file-confirmation.component';
import { JobStatusCard } from '../../job-status-card/job-status-card.component';
import { ImportFailedJob } from '../import-failed-job/import-failed-job.component';

type StagedImportDatasetProps = {
    message: string;
    fileName: string;
    stagedDatasetId: string;
    primaryButtonLabel: string;
    onOpen: () => void;
    deleteEntry: () => void;
};

export const StagedImportDataset = ({
    message,
    fileName,
    stagedDatasetId,
    primaryButtonLabel,
    onOpen,
    deleteEntry,
}: StagedImportDatasetProps) => {
    const { t } = useTranslation();
    const { error, isError, isFetching, data: stagedDataset } = useStagedDataset(stagedDatasetId);

    if (isError) {
        return (
            <ImportFailedJob
                size={0}
                fileName={fileName}
                error={getErrorMessage(error)}
                message={t('dataset.import.stagedFileReadError')}
                stagedDatasetId={stagedDatasetId}
                deleteEntry={deleteEntry}
            />
        );
    }

    return (
        <JobStatusCard
            title={t('dataset.import.jobTitle', { fileName, size: formatBytes(stagedDataset?.size ?? 0) })}
            actionButtons={
                <>
                    <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} deleteEntry={deleteEntry} />
                    <Button onPress={onOpen} isPending={isFetching} isDisabled={isFetching}>
                        {primaryButtonLabel}
                    </Button>
                </>
            }
            bottomIcon={<InfoOutline width={16} height={16} />}
            bottomIconMessage={message}
        />
    );
};
