// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { EnablePipelineBlockedDialog } from '@/components/enable-pipeline-blocked-dialog/enable-pipeline-blocked-dialog.component';
import { toast } from '@/components/toast/toast.component';
import { Switch } from '@geti-ui/ui';
import { useDisablePipeline, useEnablePipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useWebRTCConnection } from '../stream/web-rtc-connection-provider';

export const TogglePipelineButton = () => {
    const projectId = useProjectIdentifier();
    const [isEnableBlockedDialogOpen, setIsEnableBlockedDialogOpen] = useState(false);

    const pipelineQuery = usePipeline();
    const disablePipelineMutation = useDisablePipeline();
    const enablePipelineMutation = useEnablePipeline();
    const { stop: stopStream, status: streamStatus } = useWebRTCConnection();

    const isPipelineEnabled = pipelineQuery.data?.status === 'running';
    const canEnablePipeline = useIsPipelineConfigured(pipelineQuery.data);
    const isPending = disablePipelineMutation.isPending || enablePipelineMutation.isPending;

    const mutationParams = { params: { path: { project_id: projectId } } };

    const handleToggle = () => {
        const mutationOptions = {
            onSuccess: () => {
                toast({
                    type: 'success',
                    message: `Pipeline ${isPipelineEnabled ? 'disabled' : 'enabled'} successfully`,
                });

                if (isPipelineEnabled && streamStatus !== 'idle' && streamStatus !== 'failed') {
                    void stopStream();
                }
            },
        };

        if (isPipelineEnabled) {
            disablePipelineMutation.mutate(mutationParams, mutationOptions);
        } else {
            if (!canEnablePipeline) {
                setIsEnableBlockedDialogOpen(true);

                return;
            }

            enablePipelineMutation.mutate(mutationParams, mutationOptions);
        }
    };

    return (
        <>
            <Switch isEmphasized isSelected={isPipelineEnabled} isDisabled={isPending} onChange={handleToggle}>
                Pipeline {isPipelineEnabled ? 'enabled' : 'disabled'}
            </Switch>

            <EnablePipelineBlockedDialog
                isOpen={isEnableBlockedDialogOpen}
                onClose={() => setIsEnableBlockedDialogOpen(false)}
            />
        </>
    );
};
