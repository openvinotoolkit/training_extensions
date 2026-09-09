// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ConfidenceThreshold } from '@/components/confidence-threshold/confidence-threshold.component';

import { usePredictionSetup } from '../../../../annotator/predictions-setup-provider.component';

export const PredictionConfidenceThreshold = () => {
    const { selectedModel, confidenceThreshold, changeConfidenceThreshold } = usePredictionSetup();

    const defaultValue = selectedModel?.optimalConfidenceThreshold ?? null;

    // Models whose task does not use a confidence threshold have no optimal value
    if (defaultValue === null || confidenceThreshold === null) {
        return null;
    }

    return (
        <ConfidenceThreshold
            value={confidenceThreshold}
            defaultValue={defaultValue}
            onChange={changeConfidenceThreshold}
        />
    );
};
