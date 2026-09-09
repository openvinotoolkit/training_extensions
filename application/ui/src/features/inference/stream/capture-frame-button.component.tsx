// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
import { Button } from '@geti-ui/ui';
import { useCapturePipelineFrame, usePipeline, usePipelineHealth } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useWebRTCConnection } from './web-rtc-connection-provider';

export const CaptureFrameButton = () => {
    const projectId = useProjectIdentifier();
    const { data: pipeline } = usePipeline();
    const { data: pipelineHealth } = usePipelineHealth();
    const { status: streamStatus } = useWebRTCConnection();
    const captureFrameMutation = useCapturePipelineFrame();

    const isPipelineRunning = pipeline.status === 'running';
    const isStreamConnected = streamStatus === 'connected';
    // The backend collects the *next* frame the source produces, so a source that ended
    // (e.g. a non-looping video file) or errored can never fulfil the request.
    const sourceStatus = pipelineHealth?.components?.source.status;
    const isSourceExhausted = sourceStatus !== undefined && sourceStatus !== 'ok';

    const isCaptureDisabled =
        !isPipelineRunning || !isStreamConnected || isSourceExhausted || captureFrameMutation.isPending;

    const handleCapture = () => {
        captureFrameMutation.mutate(
            { params: { path: { project_id: projectId } } },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: 'Frame captured and added to the dataset.' });
                },
            }
        );
    };

    return (
        <Button
            variant={'secondary'}
            onPress={handleCapture}
            isDisabled={isCaptureDisabled}
            isPending={captureFrameMutation.isPending}
        >
            Capture frame
        </Button>
    );
};
