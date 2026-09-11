// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StagedImportDataset } from '@/components/import-card-status/staged-import-dataset/staged-import-dataset.component';
import { LoadingImportDataset } from '@/components/loading-import-dataset/loading-import-dataset.component';
import { PrepareImportDataset } from '@/components/prepare-import-dataset/prepare-import-dataset.component';
import { Flex } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useImportDatasetAsNewProject } from 'hooks/storage/use-import-dataset-as-new-project.hook';
import { isEmpty, partition } from 'lodash-es';

import { getQueryKey } from '../../../../query-client/query-client';
import { ImportDatasetAsNewProjectState } from '../../../dataset/import-export/import-dataset/util';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';

export const ImportJobsList = () => {
    const queryClient = useQueryClient();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialog();
    const { getAllImportEntries, deleteImportEntry, updateImportEntryStep } = useImportDatasetAsNewProject();

    const importEntries = getAllImportEntries();

    const [preparingImports, others] = partition(importEntries, ({ step }) => step === 'preparing');
    const [taskTypeSelectionImports, restItems] = partition(others, ({ step }) => step === 'taskTypeSelection');
    const [labelMappingImports, leftOvers] = partition(restItems, ({ step }) => step === 'labelMapping');
    const [importingJob] = partition(leftOvers, ({ step }) => step === 'importing');

    const importingJobQueue = importingJob.reverse();
    const preparingImportsQueue = preparingImports.reverse();
    const labelMappingImportsQueue = labelMappingImports.reverse();
    const taskTypeSelectionImportsQueue = taskTypeSelectionImports.reverse();

    const handleImportSuccess = async () => {
        await queryClient.invalidateQueries({
            queryKey: getQueryKey(['get', '/api/projects']),
        });
    };

    const handleOpen = (openState: ImportDatasetAsNewProjectState, stagedDatasetId: string) => {
        setCurrentStep(openState);
        setCurrentStagedId(stagedDatasetId);
        datasetImportDialogState.open();
    };

    if (
        isEmpty(preparingImportsQueue) &&
        isEmpty(taskTypeSelectionImportsQueue) &&
        isEmpty(labelMappingImportsQueue) &&
        isEmpty(importingJobQueue)
    ) {
        return null;
    }

    return (
        <Flex
            gap='size-250'
            maxHeight='228px'
            direction='column'
            marginBottom='size-250'
            UNSAFE_style={{ overflowY: 'auto' }}
        >
            {preparingImportsQueue.map(({ size, fileName, stagedDatasetId, prepareJobId }) => (
                <PrepareImportDataset
                    key={`prepare-${stagedDatasetId}`}
                    size={size}
                    fileName={fileName}
                    jobId={prepareJobId}
                    stagedDatasetId={stagedDatasetId}
                    onSuccess={() => updateImportEntryStep(stagedDatasetId, 'taskTypeSelection')}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}

            {taskTypeSelectionImportsQueue.map(({ fileName, stagedDatasetId }) => (
                <StagedImportDataset
                    key={`task-type-${stagedDatasetId}`}
                    fileName={fileName}
                    message={'Select task type'}
                    stagedDatasetId={stagedDatasetId}
                    onOpen={() => handleOpen('taskTypeSelection', stagedDatasetId)}
                    primaryButtonLabel={'Select task type'}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}

            {labelMappingImportsQueue.map(({ fileName, stagedDatasetId }) => (
                <StagedImportDataset
                    key={`label-mapping-${stagedDatasetId}`}
                    fileName={fileName}
                    message={'Map labels for the uploaded dataset'}
                    stagedDatasetId={stagedDatasetId}
                    onOpen={() => handleOpen('labelMapping', stagedDatasetId)}
                    primaryButtonLabel={'Map labels'}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}

            {importingJobQueue.map(({ size, fileName, stagedDatasetId, importJobId }) => (
                <LoadingImportDataset
                    key={`loading-${stagedDatasetId}`}
                    size={size}
                    fileName={fileName}
                    jobId={String(importJobId)}
                    stagedDatasetId={String(stagedDatasetId)}
                    onSuccess={handleImportSuccess}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}
        </Flex>
    );
};
