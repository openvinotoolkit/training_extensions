// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { InferenceDevices } from '@/components/inference-devices/inference-devices.component';
import { Loading } from '@geti-ui/ui';

import { usePredictionSetup } from '../../../../annotator/predictions-setup-provider.component';

type PredictionInferenceDevicesProps = {
    isDisabled?: boolean;
};

export const PredictionInferenceDevices = ({ isDisabled }: PredictionInferenceDevicesProps) => {
    const { selectedDevice, changeSelectedDevice } = usePredictionSetup();

    return (
        <Suspense fallback={<Loading mode={'inline'} />}>
            <InferenceDevices
                label={'Inference device'}
                selectedKey={selectedDevice}
                onSelectionChange={changeSelectedDevice}
                isDisabled={isDisabled}
            />
        </Suspense>
    );
};
