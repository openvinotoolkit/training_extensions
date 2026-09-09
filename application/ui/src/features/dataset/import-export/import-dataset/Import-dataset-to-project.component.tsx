// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FileUploadedResponse, ImportUploadFile } from '@/components/import-upload-file/import-upload-file.component';
import { getFormatOptions } from '@/components/util';
import { Content, Dialog, DialogContainer, Divider, Heading } from '@geti-ui/ui';
import { useProject } from 'hooks/api/project.hook';
import { useImportDatasetToProject } from 'hooks/storage/use-import-dataset-to-project.hook';

import { isNonEmptyString } from '../../../../shared/util';
import { useImportDatasetDialogState } from '../../providers/export-import-dataset-dialog-provider.component';
import { ImportDatasetButtons } from './import-dataset-buttons/import-dataset-buttons.component';
import { ImportProcess } from './import-process/import-process.component';
import { LabelMapping } from './label-mapping/label-mapping.component';

export const ImportDatasetToProject = () => {
    const { data: selectedProject } = useProject();

    const { appendImportEntry } = useImportDatasetToProject();
    const { datasetImportDialogState, currentStep, currentStagedId, setCurrentStep, setCurrentStagedId } =
        useImportDatasetDialogState();

    const formatOptions = getFormatOptions(selectedProject.task.task_type)
        .map(({ label }) => label)
        .join(', ');

    const handleFileUploaded = (response: FileUploadedResponse) => {
        setCurrentStep('preparing');
        setCurrentStagedId(response.stagedDatasetId);
        appendImportEntry({ ...response, step: 'preparing', importJobId: null });
    };

    return (
        <DialogContainer onDismiss={datasetImportDialogState.close}>
            {datasetImportDialogState.isOpen && (
                <Dialog aria-label={'Import dataset to project'} width={800}>
                    <Heading>Import dataset</Heading>
                    <Divider />
                    <Content
                        minHeight={'size-5000'}
                        UNSAFE_style={{ background: 'var(--spectrum-global-color-gray-50)' }}
                    >
                        {currentStep === 'uploading' && (
                            <ImportUploadFile formatOptions={formatOptions} onFileUploaded={handleFileUploaded} />
                        )}

                        {currentStep === 'preparing' && isNonEmptyString(currentStagedId) && (
                            <ImportProcess currentStagedId={currentStagedId} />
                        )}

                        {currentStep === 'labelMapping' && isNonEmptyString(currentStagedId) && (
                            <LabelMapping stagedDatasetId={currentStagedId} />
                        )}
                    </Content>

                    <ImportDatasetButtons
                        currentStep={currentStep}
                        stagedDatasetId={currentStagedId}
                        onClose={datasetImportDialogState.close}
                    />
                </Dialog>
            )}
        </DialogContainer>
    );
};
