// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
import { Key } from '@geti-ui/ui';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';

import { useDisablePipeline, useEnablePipeline, useProjectPipeline } from '../../../../hooks/api/pipeline.hook';

const PROJECT_ACTIONS = { rename: 'Rename', delete: 'Delete' };

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
    const enablePipelineMutation = useEnablePipeline();
    const disablePipelineMutation = useDisablePipeline();
    const projectPipelineQuery = useProjectPipeline(projectId);

    const isPipelineConfigured = useIsPipelineConfigured(projectPipelineQuery.data);

    const menuActions: MenuAction[] = [
        ...(isPipelineRunning
            ? [{ key: 'disable-pipeline', label: 'Disable pipeline' }]
            : [{ key: 'enable-pipeline', label: 'Enable pipeline' }]),
        ...Object.entries(PROJECT_ACTIONS).map(([key, label]) => ({ key, label })),
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
                        toast({ type: 'success', message: 'Pipeline enabled successfully' });
                    },
                });
                break;
            case 'disable-pipeline':
                disablePipelineMutation.mutate(mutationParams, {
                    onSuccess: () => {
                        toast({ type: 'success', message: 'Pipeline disabled successfully' });
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
