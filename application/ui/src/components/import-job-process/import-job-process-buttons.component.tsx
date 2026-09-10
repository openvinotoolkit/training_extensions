// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, ButtonGroup } from '@geti-ui/ui';
import { useCancelJob } from 'hooks/api/jobs/jobs.hook';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

type ImportJobProcessButtonsProps = {
    prepareJobId: string;
    stagedDatasetId: string;
    onClose: () => void;
    deleteEntry: () => void;
};

export const ImportJobProcessButtons = ({
    prepareJobId,
    stagedDatasetId,
    onClose,
    deleteEntry,
}: ImportJobProcessButtonsProps) => {
    const { t } = useTranslation();
    const cancelJobMutation = useCancelJob();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose, deleteEntry });

    const isPending = cancelJobMutation.isPending || deleteFileMutation.isPending;

    const handleCancelJob = async (jobId: string) => {
        await cancelJobMutation.mutateAsync({ params: { path: { job_id: jobId } } });
        deleteFileMutation.mutate();
    };

    return (
        <ButtonGroup>
            <Button
                variant='negative'
                isPending={isPending}
                isDisabled={isPending}
                onPress={() => handleCancelJob(prepareJobId)}
            >
                {t('dataset.import.cancel')}
            </Button>
            <Button onPress={onClose} variant='secondary' isPending={isPending} isDisabled={isPending}>
                {t('dataset.import.hide')}
            </Button>
        </ButtonGroup>
    );
};
