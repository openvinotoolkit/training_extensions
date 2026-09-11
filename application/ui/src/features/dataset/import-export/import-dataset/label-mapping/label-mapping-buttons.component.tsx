// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, ButtonGroup } from '@geti-ui/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';
import { useImportDatasetToProject } from 'hooks/storage/use-import-dataset-to-project.hook';

import { IMPORT_DATASET_FORM_ID } from './util';

type LabelMappingButtonsProps = {
    stagedDatasetId: string;
    onClose: () => void;
};

export const LabelMappingButtons = ({ stagedDatasetId, onClose }: LabelMappingButtonsProps) => {
    const { t } = useTranslation();
    const { deleteImportEntry } = useImportDatasetToProject();
    const deleteFileMutation = useDeleteStagedDataset({
        stagedDatasetId,
        onSuccess: onClose,
        deleteEntry: () => deleteImportEntry(stagedDatasetId),
    });

    const handleDelete = () => {
        deleteFileMutation.mutate();
    };

    return (
        <ButtonGroup>
            <Button
                onPress={handleDelete}
                variant='negative'
                isPending={deleteFileMutation.isPending}
                isDisabled={deleteFileMutation.isPending}
            >
                {t('dataset.import.deleteStagedFile.delete')}
            </Button>

            <Button onPress={onClose} variant='secondary'>
                {t('dataset.import.hide')}
            </Button>

            <Button type='submit' form={IMPORT_DATASET_FORM_ID} variant='accent'>
                {t('dataset.import.labelMapping.submit')}
            </Button>
        </ButtonGroup>
    );
};
