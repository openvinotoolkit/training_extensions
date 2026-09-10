// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { toast } from '@/components/toast/toast.component';
import { useTranslation, type TranslateFn } from '@/i18n';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isFunction } from 'lodash-es';

import { getQueryKey } from '../../../query-client/query-client';

const toastId = 'deleting-notification';

const useDeleteMediaItemsMutation = (t: TranslateFn) => {
    return $api.useMutation('delete', `/api/projects/{project_id}/dataset/media`, {
        meta: { error: { notify: () => false } },
        onError: (error) => {
            toast({
                id: toastId,
                type: 'error',
                message: t('dataset.delete.deleteError', { detail: error?.detail }),
            });
        },
    });
};

export const useDeleteMediaItem = () => {
    const { t } = useTranslation();
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();
    const deleteItemsMutation = useDeleteMediaItemsMutation(t);

    const alertDialogState = useOverlayTriggerState({});

    const handleDeleteItems = async (media_ids: string[], onDeleted?: (ids: string[]) => void) => {
        alertDialogState.close();

        toast({ id: toastId, type: 'info', message: t('dataset.delete.deletingItems') });

        deleteItemsMutation.mutate(
            {
                body: { media_ids },
                params: { path: { project_id: projectId } },
            },
            { onSuccess: () => handleSuccess(media_ids, onDeleted) }
        );
    };

    const handleSuccess = (deletedIds: string[], onDeleted?: (ids: string[]) => void) => {
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/statistics',
                { params: { path: { project_id: projectId } } },
            ]),
        });
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                { params: { path: { project_id: projectId } } },
            ]),
        });

        isFunction(onDeleted) && onDeleted(deletedIds);

        toast({
            id: toastId,
            type: 'success',
            message: t('dataset.delete.deletedSuccess', { count: deletedIds.length }),
            duration: 3000,
        });
    };

    return {
        deleteMedia: handleDeleteItems,
        isPending: deleteItemsMutation.isPending,
        isDeleteDialogOpen: alertDialogState.isOpen,
        openDeleteDialog: alertDialogState.open,
        closeDeleteDialog: alertDialogState.close,
    };
};
