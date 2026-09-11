// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ImportJobProcessButtons } from '@/components/import-job-process/import-job-process-buttons.component';
import { Button, ButtonGroup } from '@geti-ui/ui';
import { useImportDatasetAsNewProject } from 'hooks/storage/use-import-dataset-as-new-project.hook';

import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';
import { ImportLabelMappingButtons } from '../import-label-mapping/import-label-mapping-buttons.component';
import { ImportTaskSelectionButtons } from '../import-task-selection/import-task-selection-buttons.component';

type ImportDatasetButtonsProps = {
    onClose: () => void;
    stagedDatasetId: string | null;
    currentStep: ImportDatasetAsNewProjectState;
};

export const ImportDatasetButtons = ({ currentStep, stagedDatasetId, onClose }: ImportDatasetButtonsProps) => {
    const { getImportEntry, deleteImportEntry } = useImportDatasetAsNewProject();
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

    if (currentStep === 'taskTypeSelection') {
        return (
            <ImportTaskSelectionButtons
                onClose={onClose}
                stagedDatasetId={stagedDatasetId}
                deleteEntry={() => deleteImportEntry(stagedDatasetId)}
            />
        );
    }

    if (currentStep === 'labelMapping') {
        return (
            <ImportLabelMappingButtons
                onClose={onClose}
                stagedDatasetId={stagedDatasetId}
                deleteEntry={() => deleteImportEntry(stagedDatasetId)}
            />
        );
    }

    return (
        <ButtonGroup>
            <Button onPress={onClose} variant='secondary'>
                Cancel
            </Button>
        </ButtonGroup>
    );
};
