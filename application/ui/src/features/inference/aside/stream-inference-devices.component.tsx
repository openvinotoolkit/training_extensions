// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { InferenceDevices } from '@/components/inference-devices/inference-devices.component';
import { usePatchPipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

export const StreamInferenceDevices = () => {
    const { data: pipeline } = usePipeline();
    const projectId = useProjectIdentifier();
    const [selectedKey, setSelectedKey] = useState<string>(pipeline.device);
    const updatePipeline = usePatchPipeline();

    const handleChange = (device: string) => {
        setSelectedKey(device);
        updatePipeline.mutate(
            {
                params: { path: { project_id: projectId } },
                body: { device },
            },
            {
                onError: (error) => {
                    if (error) {
                        setSelectedKey(pipeline.device);
                    }
                },
            }
        );
    };

    return <InferenceDevices label={'Inference device'} selectedKey={selectedKey} onSelectionChange={handleChange} />;
};
