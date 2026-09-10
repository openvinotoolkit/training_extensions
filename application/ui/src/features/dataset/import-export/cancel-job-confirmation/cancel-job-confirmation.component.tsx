// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { useTranslation } from '@/i18n';
import { AlertDialog, Button, DialogTrigger } from '@geti-ui/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { isInvalidJob } from 'hooks/api/util';

type CancelJobConfirmationProps = {
    jobId: string;
    onRemove: () => void | Promise<void>;
};

export const CancelJobConfirmation = ({ jobId, onRemove }: CancelJobConfirmationProps) => {
    const { t } = useTranslation();
    const dialogState = useOverlayTriggerState({});
    const cancelMutation = $api.useMutation('post', `/api/jobs/{job_id}:cancel`);

    const handleCancel = () => {
        cancelMutation.mutate(
            { params: { path: { job_id: jobId } } },
            {
                onSuccess: async () => await onRemove(),
                onError: async (error) => {
                    isInvalidJob(error) && (await onRemove());
                },
                onSettled: () => {
                    dialogState.close();
                },
            }
        );
    };

    return (
        <DialogTrigger>
            <Button
                variant='negative'
                style='outline'
                aria-label='cancel job dialog'
                isDisabled={cancelMutation.isPending}
                isPending={cancelMutation.isPending}
            >
                {t('dataset.jobs.cancel.trigger')}
            </Button>
            <AlertDialog
                title={t('dataset.jobs.cancel.title')}
                variant='destructive'
                cancelLabel={t('dataset.jobs.cancel.dismiss')}
                autoFocusButton='primary'
                primaryActionLabel={t('dataset.jobs.cancel.confirm')}
                onPrimaryAction={handleCancel}
                onSecondaryAction={dialogState.close}
                isPrimaryActionDisabled={cancelMutation.isPending}
            >
                {t('dataset.jobs.cancel.confirmation', { jobId })}
            </AlertDialog>
        </DialogTrigger>
    );
};
