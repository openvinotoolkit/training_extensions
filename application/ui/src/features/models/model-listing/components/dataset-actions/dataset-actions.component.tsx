// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ExportDatasetConfig } from '@/components/export-dataset-config-dialog/export-dataset-config.component';
import { ActionButton, AlertDialog, DialogContainer, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useDeleteDatasetRevision } from '../../hooks/use-delete-dataset-revision.hook';
import { useRenameDatasetRevision } from '../../hooks/use-rename-dataset-revision.hook';
import type { DatasetGroup } from '../../types';
import { RenameDatasetRevisionDialog } from '../rename-dataset-revision-dialog.component';
import { DatasetRevisionStatistics } from './dataset-revision-statistics/dataset-revision-statistics.component';

type DatasetActionsProps = {
    dataset: DatasetGroup;
};

export const DatasetActions = ({ dataset }: DatasetActionsProps) => {
    const projectId = useProjectIdentifier();
    const renameDatasetRevisionMutation = useRenameDatasetRevision();
    const deleteDatasetRevisionMutation = useDeleteDatasetRevision();

    const renameDialog = useOverlayTriggerState({});
    const deleteDialog = useOverlayTriggerState({});
    const exportDialog = useOverlayTriggerState({});

    const handleDatasetMenuAction = (key: Key) => {
        switch (key) {
            case 'rename':
                renameDialog.open();
                break;
            case 'delete':
                deleteDialog.open();
                break;
            case 'export':
                exportDialog.open();
                break;
            default:
                break;
        }
    };

    const handleRename = (newName: string) => {
        renameDatasetRevisionMutation.mutate(
            {
                params: { path: { project_id: projectId, dataset_revision_id: dataset.id } },
                body: { name: newName },
            },
            {
                onSuccess: () => {
                    renameDialog.close();
                },
            }
        );
    };

    const handleDelete = () => {
        deleteDatasetRevisionMutation.mutate({
            params: { path: { project_id: projectId, dataset_revision_id: dataset.id } },
        });
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Dataset actions'}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleDatasetMenuAction} aria-label={'Dataset actions menu'}>
                    <Item key={'rename'}>Rename</Item>
                    <Item key={'delete'}>Delete</Item>
                    <Item key={'export'}>Export</Item>
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={renameDialog.close}>
                {renameDialog.isOpen && (
                    <RenameDatasetRevisionDialog
                        currentName={dataset.name}
                        onRename={handleRename}
                        isPending={renameDatasetRevisionMutation.isPending}
                        onClose={renameDialog.close}
                    />
                )}
            </DialogContainer>

            <DialogContainer onDismiss={deleteDialog.close}>
                {deleteDialog.isOpen && (
                    <AlertDialog
                        title='Delete dataset revision'
                        variant='destructive'
                        primaryActionLabel='Delete'
                        onPrimaryAction={handleDelete}
                        cancelLabel='Cancel'
                    >
                        {`Are you sure you want to delete dataset revision "${dataset.name}"? ` +
                            `You will still be able to see the model statistics but you won't ` +
                            `be able to access the training dataset files.`}
                    </AlertDialog>
                )}
            </DialogContainer>

            <ExportDatasetConfig
                name={dataset.name}
                datasetId={dataset.id}
                dialogState={exportDialog}
                statistics={<DatasetRevisionStatistics datasetRevisionId={dataset.id} />}
            />
        </>
    );
};
