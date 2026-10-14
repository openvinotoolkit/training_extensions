// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useTranslation } from '@/i18n';
import { ActionButton, AlertDialog, DialogContainer, Item, Menu, MenuTrigger, type Key } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useCancelJob } from 'hooks/api/jobs/jobs.hook';

import type { ModelJob } from '../../utils';

const ACTIONS = {
    LOGS: 'logs',
    CANCEL: 'cancel',
};

type JobActionsMenuProps = {
    job: ModelJob;
    onViewLogs?: () => void;
    canCancel: boolean;
};

export const JobActionsMenu = ({ job, onViewLogs, canCancel }: JobActionsMenuProps) => {
    const { t } = useTranslation();
    const cancelJobMutation = useCancelJob();
    const [isConfirmOpen, setIsConfirmOpen] = useState(false);

    const handleAction = (key: Key) => {
        if (key === ACTIONS.LOGS) onViewLogs?.();
        if (key === ACTIONS.CANCEL) setIsConfirmOpen(true);
    };

    const handleCancel = () => {
        cancelJobMutation.mutate(
            { params: { path: { job_id: job.job_id } } },
            { onSuccess: () => setIsConfirmOpen(false) }
        );
    };

    if (!onViewLogs && !canCancel) return null;

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={`Job actions for ${job.metadata.model.name}`}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} aria-label={'Job actions menu'}>
                    {[
                        ...(onViewLogs ? [<Item key={ACTIONS.LOGS}>{t('jobs.actions.viewLogs')}</Item>] : []),
                        ...(canCancel ? [<Item key={ACTIONS.CANCEL}>{t('common.actions.cancel')}</Item>] : []),
                    ]}
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={() => setIsConfirmOpen(false)}>
                {isConfirmOpen && (
                    <AlertDialog
                        title={t('jobs.cancel.title')}
                        variant='destructive'
                        primaryActionLabel={t('jobs.cancel.title')}
                        onPrimaryAction={handleCancel}
                        isPrimaryActionDisabled={cancelJobMutation.isPending}
                        cancelLabel={t('common.actions.dismiss')}
                    >
                        {t('jobs.cancel.confirmation', { jobId: job.job_id })}
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};
