// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StagedImportDataset } from '@/components/import-card-status/staged-import-dataset/staged-import-dataset.component';
import { LoadingImportDataset } from '@/components/loading-import-dataset/loading-import-dataset.component';
import { PrepareImportDataset } from '@/components/prepare-import-dataset/prepare-import-dataset.component';
import { Flex } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useImportDatasetToProject } from 'hooks/storage/use-import-dataset-to-project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty, partition } from 'lodash-es';

import { getQueryKey } from '../../../../query-client/query-client';
import { useImportDatasetDialogState } from '../../providers/export-import-dataset-dialog-provider.component';

export const ImportJobsList = () => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialogState();
    const { getAllImportEntries, deleteImportEntry, updateImportEntryStep } = useImportDatasetToProject();

    const importEntries = getAllImportEntries();

    const [preparingImports, otherItems] = partition(importEntries, ({ step }) => step === 'preparing');
    const [stagedImports, loadingItems] = partition(otherItems, ({ step }) => step === 'labelMapping');

    const loadingItemsQueue = loadingItems.reverse();
    const stagedImportsQueue = stagedImports.reverse();
    const preparingImportsQueue = preparingImports.reverse();

    const handleImportSuccess = async () => {
        const params = { path: { project_id: projectId } };

        await Promise.all([
            queryClient.invalidateQueries({
                queryKey: getQueryKey(['get', '/api/projects/{project_id}/dataset/media', { params }]),
            }),
            queryClient.invalidateQueries({
                queryKey: getQueryKey(['get', '/api/projects/{project_id}/dataset/items', { params }]),
            }),
        ]);
    };

    const handleOpen = (stagedDatasetId: string) => {
        setCurrentStep('labelMapping');
        setCurrentStagedId(stagedDatasetId);
        datasetImportDialogState.open();
    };

    if (isEmpty(preparingImportsQueue) && isEmpty(stagedImportsQueue) && isEmpty(loadingItemsQueue)) {
        return null;
    }

    return (
        <Flex
            gap='size-250'
            direction='column'
            maxHeight='size-3400'
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
                    onSuccess={() => updateImportEntryStep(stagedDatasetId, 'labelMapping')}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}

            {stagedImportsQueue.map(({ fileName, stagedDatasetId }) => (
                <StagedImportDataset
                    key={`staged-${stagedDatasetId}`}
                    fileName={fileName}
                    message={'Map labels for the uploaded dataset'}
                    stagedDatasetId={stagedDatasetId}
                    primaryButtonLabel={'Continue'}
                    onOpen={() => handleOpen(stagedDatasetId)}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}

            {loadingItemsQueue.map(({ size, fileName, stagedDatasetId, importJobId }) => (
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
