// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, DialogContainer } from '@geti-ui/ui';
import { useDeleteProject } from 'hooks/api/project.hook';
import { useTranslation } from 'react-i18next';

import { toast } from '../toast/toast.component';

type DeleteProjectDialogProps = {
    isOpen: boolean;
    projectId: string;
    projectName: string;
    onClose: () => void;
    onDeleted?: () => void;
};

export const DeleteProjectDialog = ({
    isOpen,
    projectId,
    projectName,
    onClose,
    onDeleted,
}: DeleteProjectDialogProps) => {
    const { t } = useTranslation();
    const deleteMutation = useDeleteProject();

    const handleDelete = () => {
        deleteMutation.mutate(
            { params: { path: { project_id: projectId } } },
            {
                onSuccess: () => {
                    onDeleted?.();
                    toast({ type: 'success', message: t('project.delete.success') });
                },
            }
        );
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <AlertDialog
                    title={t('project.delete.title')}
                    variant='destructive'
                    cancelLabel={t('common.actions.cancel')}
                    primaryActionLabel={t('common.actions.delete')}
                    onPrimaryAction={handleDelete}
                    onSecondaryAction={onClose}
                    autoFocusButton='primary'
                    isPrimaryActionDisabled={deleteMutation.isPending}
                >
                    {t('project.delete.confirmation', { projectName })}
                </AlertDialog>
            )}
        </DialogContainer>
    );
};
