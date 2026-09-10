// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { AlertDialog, Content, Text } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { datasetViewsQueryOptions } from './api/use-dataset-views';
import { useDeleteDatasetViewMutation } from './api/use-delete-dataset-view';
import { DatasetView } from './type';

type DeleteDatasetViewDialogProps = {
    datasetView: DatasetView;
    onSuccess: () => void;
    onCancel: () => void;
};

const useDeleteDatasetView = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const deleteDatasetViewMutation = useDeleteDatasetViewMutation();

    const deleteDatasetView = ({ datasetViewId, onSuccess }: { datasetViewId: string; onSuccess: () => void }) => {
        deleteDatasetViewMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                        dataset_view_id: datasetViewId,
                    },
                },
            },
            {
                onSuccess: async () => {
                    await queryClient.invalidateQueries({ queryKey: datasetViewsQueryOptions(projectId).queryKey });
                    onSuccess();
                },
            }
        );
    };

    return {
        deleteDatasetView,
        isPending: deleteDatasetViewMutation.isPending,
    };
};

export const DeleteDatasetViewDialog = ({ datasetView, onSuccess, onCancel }: DeleteDatasetViewDialogProps) => {
    const { t } = useTranslation();
    const { deleteDatasetView, isPending } = useDeleteDatasetView();

    const deleteView = () => {
        deleteDatasetView({ datasetViewId: datasetView.id, onSuccess });
    };

    return (
        <AlertDialog
            title={t('dataset.views.deleteTitle')}
            primaryActionLabel={t('dataset.views.delete')}
            variant={'destructive'}
            onPrimaryAction={deleteView}
            onCancel={onCancel}
            secondaryActionLabel={t('dataset.views.close')}
            isPrimaryActionDisabled={isPending}
        >
            <Content>
                <Text>{t('dataset.views.deleteConfirmation', { viewName: datasetView.name })}</Text>
            </Content>
        </AlertDialog>
    );
};
