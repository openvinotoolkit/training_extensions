// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ImportJobProcess } from '@/components/import-job-process/import-job-process.component';

import { useImportDatasetToProject } from '../../../../../hooks/storage/use-import-dataset-to-project.hook';
import { useImportDatasetDialogState } from '../../../providers/export-import-dataset-dialog-provider.component';

type ImportProcessProps = {
    currentStagedId: string;
};

export const ImportProcess = ({ currentStagedId }: ImportProcessProps) => {
    const { setCurrentStep } = useImportDatasetDialogState();
    const { getImportEntry, updateImportEntryStep, deleteImportEntry } = useImportDatasetToProject();
    const importLsEntry = getImportEntry(currentStagedId);

    return (
        <ImportJobProcess
            jobId={importLsEntry?.prepareJobId}
            fileName={importLsEntry?.fileName ?? ''}
            message='Scanning and analyzing the dataset archive to import...'
            onError={() => {
                setCurrentStep('uploading');
                deleteImportEntry(currentStagedId);
            }}
            onSuccess={() => {
                setCurrentStep('labelMapping');
                updateImportEntryStep(currentStagedId, 'labelMapping');
            }}
        />
    );
};
