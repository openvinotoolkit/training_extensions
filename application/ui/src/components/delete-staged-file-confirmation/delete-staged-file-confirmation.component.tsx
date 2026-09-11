// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { AlertDialog, Button, DialogTrigger } from '@geti-ui/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

type DeleteStagedFileConfirmationProps = {
    stagedDatasetId: string;
    deleteEntry: () => void;
};

export const DeleteStagedFileConfirmation = ({ stagedDatasetId, deleteEntry }: DeleteStagedFileConfirmationProps) => {
    const { t } = useTranslation();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const handleCancel = () => {
        deleteFileMutation.mutate();
    };

    return (
        <DialogTrigger>
            <Button variant='secondary' style='fill' aria-label='delete import dataset status'>
                {t('dataset.import.deleteStagedFile.delete')}
            </Button>
            <AlertDialog
                title={t('dataset.import.deleteStagedFile.title')}
                variant='destructive'
                cancelLabel={t('dataset.import.deleteStagedFile.cancel')}
                autoFocusButton='primary'
                primaryActionLabel={t('dataset.import.deleteStagedFile.delete')}
                onPrimaryAction={handleCancel}
                isPrimaryActionDisabled={deleteFileMutation.isPending}
            >
                {t('dataset.import.deleteStagedFile.confirmation', { stagedDatasetId })}
            </AlertDialog>
        </DialogTrigger>
    );
};
