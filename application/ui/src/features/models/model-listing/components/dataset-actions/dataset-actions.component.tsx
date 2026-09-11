// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ExportDatasetConfig } from '@/components/export-dataset-config-dialog/export-dataset-config.component';
import { useTranslation } from '@/i18n';
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
    const { t } = useTranslation();
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
                    <Item key={'rename'}>{t('dataset.revisions.actions.rename')}</Item>
                    <Item key={'delete'}>{t('dataset.revisions.actions.delete')}</Item>
                    <Item key={'export'}>{t('dataset.revisions.actions.export')}</Item>
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
                        title={t('dataset.revisions.delete.title')}
                        variant='destructive'
                        primaryActionLabel={t('dataset.revisions.actions.delete')}
                        onPrimaryAction={handleDelete}
                        cancelLabel={t('dataset.revisions.actions.cancel')}
                    >
                        {t('dataset.revisions.delete.description', { name: dataset.name })}
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
