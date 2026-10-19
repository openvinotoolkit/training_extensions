// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { PredictionInferenceDevices } from './prediction-inference-devices.component';

const mockDevices = [
    { type: 'cpu' as const, name: 'CPU', memory: null, index: null },
    { type: 'xpu' as const, name: 'XPU', memory: null, index: null },
];

const mockChangeSelectedDevice = rs.fn();
const mockSelectedDevice = 'cpu';

rs.mock('../../../../../modules/annotator/predictions-setup-provider.component', () => ({
    usePredictionSetup: () => ({
        selectedDevice: mockSelectedDevice,
        changeSelectedDevice: mockChangeSelectedDevice,
        selectableModels: [],
        selectedModelId: null,
        selectedModel: undefined,
        changeSelectedModelId: rs.fn(),
    }),
}));

describe('PredictionInferenceDevices', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        server.use(http.get('/api/system/devices/inference', () => HttpResponse.json(mockDevices)));
    });

    it('renders InferenceDevices with selectedKey from context selectedDevice', async () => {
        render(<PredictionInferenceDevices />);

        // The quiet picker should show CPU as the selected device
        await waitFor(() => {
            expect(screen.getByRole('button')).toHaveTextContent('CPU');
        });
    });

    it('calls changeSelectedDevice when a different device is selected', async () => {
        render(<PredictionInferenceDevices />);

        await waitFor(() => {
            expect(screen.getByRole('button')).toHaveTextContent('CPU');
        });

        await userEvent.click(screen.getByRole('button'));
        await userEvent.click(await screen.findByRole('option', { name: /XPU/i }));

        await waitFor(() => {
            expect(mockChangeSelectedDevice).toHaveBeenCalledWith('xpu');
        });
    });
});
