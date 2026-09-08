// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { FileUploadedResponse, ImportUploadFile } from '@/components/import-upload-file/import-upload-file.component';
import { Content, Dialog, DialogContainer, Divider, Heading, View } from '@geti-ui/ui';
import { OverlayTriggerState } from '@react-stately/overlays';

import { useImportDatasetAsNewProject } from '../../../../hooks/storage/use-import-dataset-as-new-project.hook';
import { isNonEmptyString } from '../../../../shared/util';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { ImportDatasetButtons } from './import-dataset-buttons/import-dataset-buttons.component';
import { ImportErrorBoundary } from './import-error-boundary.component';
import { ImportLabelMapping } from './import-label-mapping/import-label-mapping.component';
import { ImportLoadingCard } from './import-loading-card.component';
import { ImportProcess } from './import-process/import-process.component';
import { ImportTaskSelection } from './import-task-selection/import-task-selection.component';
import { ProgressStepper } from './progress-stepper/progress-stepper.component';

import classes from './import-dataset-as-new-project.module.scss';

type ImportDatasetAsNewProjectProps = {
    dialogState: OverlayTriggerState;
};

export const ImportDatasetAsNewProject = ({ dialogState }: ImportDatasetAsNewProjectProps) => {
    const { appendImportEntry } = useImportDatasetAsNewProject();
    const { currentStagedId, setCurrentStagedId, currentStep, setCurrentStep } = useImportDatasetDialog();

    const handleFileUploaded = (response: FileUploadedResponse) => {
        appendImportEntry({ ...response, step: 'preparing', importJobId: null });
        setCurrentStep('preparing');
        setCurrentStagedId(response.stagedDatasetId);
    };

    const handleFilePrepared = () => {
        setCurrentStep('taskTypeSelection');
    };

    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <Dialog aria-label={'import-dataset-dialog'} width={860}>
                    <Heading>Create project from a dataset - Import</Heading>
                    <Divider />
                    <Content UNSAFE_className={classes.container}>
                        <ProgressStepper currentStep={currentStep} />

                        <ImportErrorBoundary>
                            <Suspense fallback={<ImportLoadingCard />}>
                                <View flex={'1'} width={'100%'} minHeight={'size-6000'} backgroundColor={'gray-50'}>
                                    {currentStep === 'uploading' && (
                                        <ImportUploadFile
                                            formatOptions='Geti, Datumaro, COCO, YOLO, VOC'
                                            onFileUploaded={handleFileUploaded}
                                        />
                                    )}

                                    {currentStep === 'preparing' && isNonEmptyString(currentStagedId) && (
                                        <ImportProcess
                                            stagedDatasetId={currentStagedId}
                                            onFilePrepared={handleFilePrepared}
                                        />
                                    )}

                                    {currentStep === 'taskTypeSelection' && isNonEmptyString(currentStagedId) && (
                                        <ImportTaskSelection stagedDatasetId={currentStagedId} />
                                    )}

                                    {currentStep === 'labelMapping' && isNonEmptyString(currentStagedId) && (
                                        <ImportLabelMapping stagedDatasetId={currentStagedId} />
                                    )}
                                </View>
                            </Suspense>
                        </ImportErrorBoundary>
                    </Content>

                    <ImportDatasetButtons
                        currentStep={currentStep}
                        stagedDatasetId={currentStagedId}
                        onClose={dialogState.close}
                    />
                </Dialog>
            )}
        </DialogContainer>
    );
};
