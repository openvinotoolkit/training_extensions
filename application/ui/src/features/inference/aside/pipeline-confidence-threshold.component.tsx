// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ConfidenceThreshold } from '@/components/confidence-threshold/confidence-threshold.component';
import { usePatchPipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

export const PipelineConfidenceThreshold = () => {
    const projectId = useProjectIdentifier();
    const { data: pipeline } = usePipeline();
    const updatePipeline = usePatchPipeline();
    const [pendingValue, setPendingValue] = useState<number | null>(null);

    const confidenceThreshold = pipeline.inference?.confidence_threshold ?? null;
    const defaultValue = pipeline.model_variant?.optimal_confidence_threshold ?? null;

    if (confidenceThreshold === null || defaultValue === null) {
        return null;
    }

    const handleChange = (value: number) => {
        setPendingValue(value);

        updatePipeline.mutate(
            {
                params: { path: { project_id: projectId } },
                body: { inference: { confidence_threshold: value } },
            },
            // Patching the pipeline refetches it, so by now the query holds either the new or the rejected value
            { onSettled: () => setPendingValue(null) }
        );
    };

    return (
        <ConfidenceThreshold
            value={pendingValue ?? confidenceThreshold}
            defaultValue={defaultValue}
            onChange={handleChange}
        />
    );
};
