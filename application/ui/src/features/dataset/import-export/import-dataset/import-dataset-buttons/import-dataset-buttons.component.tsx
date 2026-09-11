// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ImportJobProcessButtons } from '@/components/import-job-process/import-job-process-buttons.component';
import { Button, ButtonGroup } from '@geti-ui/ui';

import { useImportDatasetToProject } from '../../../../../hooks/storage/use-import-dataset-to-project.hook';
import { LabelMappingButtons } from '../label-mapping/label-mapping-buttons.component';
import { ImportDatasetToProjectState } from '../util';

type ImportDatasetButtonsProps = {
    onClose: () => void;
    stagedDatasetId: string | null;
    currentStep: ImportDatasetToProjectState;
};

export const ImportDatasetButtons = ({ currentStep, stagedDatasetId, onClose }: ImportDatasetButtonsProps) => {
    const { getImportEntry, deleteImportEntry } = useImportDatasetToProject();
    const { prepareJobId } = getImportEntry(stagedDatasetId ?? '') ?? { prepareJobId: null };

    if (stagedDatasetId === null || prepareJobId === null) {
        return (
            <ButtonGroup>
                <Button onPress={onClose} variant='secondary'>
                    Cancel
                </Button>
            </ButtonGroup>
        );
    }

    if (currentStep === 'preparing') {
        return (
            <ImportJobProcessButtons
                onClose={onClose}
                prepareJobId={prepareJobId}
                stagedDatasetId={stagedDatasetId}
                deleteEntry={() => deleteImportEntry(stagedDatasetId)}
            />
        );
    }

    if (currentStep === 'labelMapping') {
        return <LabelMappingButtons stagedDatasetId={stagedDatasetId} onClose={onClose} />;
    }

    return (
        <ButtonGroup>
            <Button onPress={onClose} variant='secondary'>
                Cancel
            </Button>
        </ButtonGroup>
    );
};
