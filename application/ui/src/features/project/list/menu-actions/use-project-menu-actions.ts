// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
import { Key } from '@geti-ui/ui';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';
import { useTranslation } from 'react-i18next';

import { useDisablePipeline, useEnablePipeline, useProjectPipeline } from '../../../../hooks/api/pipeline.hook';

type ProjectMenuCallbacks = {
    onRename: () => void;
    onDelete: () => void;
    onEnableBlocked: () => void;
};

type MenuAction = {
    key: string;
    label: string;
};

export const useProjectMenuActions = (
    projectId: string,
    callbacks: ProjectMenuCallbacks,
    isPipelineRunning?: boolean
) => {
    const { t } = useTranslation();
    const enablePipelineMutation = useEnablePipeline();
    const disablePipelineMutation = useDisablePipeline();
    const projectPipelineQuery = useProjectPipeline(projectId);

    const isPipelineConfigured = useIsPipelineConfigured(projectPipelineQuery.data);

    const menuActions: MenuAction[] = [
        ...(isPipelineRunning
            ? [{ key: 'disable-pipeline', label: t('project.list.menu.disablePipeline') }]
            : [{ key: 'enable-pipeline', label: t('project.list.menu.enablePipeline') }]),
        { key: 'rename', label: t('common.actions.rename') },
        { key: 'delete', label: t('common.actions.delete') },
    ];

    const handleAction = (key: Key) => {
        const mutationParams = { params: { path: { project_id: projectId } } };

        switch (key) {
            case 'enable-pipeline':
                if (!isPipelineConfigured) {
                    callbacks.onEnableBlocked();
                    return;
                }

                enablePipelineMutation.mutate(mutationParams, {
                    onSuccess: () => {
                        toast({ type: 'success', message: t('project.list.menu.pipelineEnabled') });
                    },
                });
                break;
            case 'disable-pipeline':
                disablePipelineMutation.mutate(mutationParams, {
                    onSuccess: () => {
                        toast({ type: 'success', message: t('project.list.menu.pipelineDisabled') });
                    },
                });
                break;
            case 'rename':
                callbacks.onRename();
                break;
            case 'delete':
                callbacks.onDelete();
                break;
            default:
                break;
        }
    };

    return {
        menuActions,
        handleAction,
    };
};
