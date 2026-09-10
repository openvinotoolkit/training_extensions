// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, ButtonGroup } from '@geti-ui/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';
import { LABEL_MAPPING_FORM_ID } from './util';

type ImportLabelMappingButtonsProps = {
    stagedDatasetId: string;
    onClose: () => void;
    deleteEntry: () => void;
};

export const ImportLabelMappingButtons = ({
    stagedDatasetId,
    onClose,
    deleteEntry,
}: ImportLabelMappingButtonsProps) => {
    const { t } = useTranslation();
    const { setCurrentStep } = useImportDatasetDialog();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, onSuccess: onClose, deleteEntry });

    const isPending = deleteFileMutation.isPending;

    const handleDeleteJob = () => {
        deleteFileMutation.mutate();
    };

    const handleBack = () => {
        setCurrentStep('taskTypeSelection');
    };

    return (
        <ButtonGroup>
            <Button variant='negative' isPending={isPending} isDisabled={isPending} onPress={handleDeleteJob}>
                {t('project.import.actions.delete')}
            </Button>

            <Button onPress={onClose} isPending={isPending} isDisabled={isPending} variant='secondary'>
                {t('project.import.actions.hide')}
            </Button>

            <Button onPress={handleBack} isPending={isPending} isDisabled={isPending} variant='secondary'>
                {t('project.import.actions.back')}
            </Button>

            <Button type='submit' variant='accent' form={LABEL_MAPPING_FORM_ID}>
                {t('project.import.actions.create')}
            </Button>
        </ButtonGroup>
    );
};
