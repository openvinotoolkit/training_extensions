// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { StreamInferenceDevices } from './stream-inference-devices.component';

const mockPipeline = getMockedPipeline({
    device: 'cpu',
});

const mockDevices = [
    { type: 'cpu' as const, name: 'CPU' },
    { type: 'xpu' as const, name: 'GPU' },
];

describe('StreamInferenceDevices', () => {
    const renderApp = (status = 200, devicesResponse = mockDevices, pipelineResponse = mockPipeline) => {
        const pipelinePatchSpy = rs.fn();

        server.use(
            http.get('/api/system/devices/inference', () => HttpResponse.json(devicesResponse)),
            http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(pipelineResponse)),
            http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                pipelinePatchSpy(await request.json());
                return HttpResponse.json(
                    {
                        project_id: '',
                        status: 'idle',
                        device: 'images_folder',
                    },
                    { status }
                );
            })
        );

        render(<StreamInferenceDevices />);

        return pipelinePatchSpy;
    };

    beforeEach(() => {
        rs.resetAllMocks();
    });

    it('displays current device selection', async () => {
        renderApp(200, mockDevices, { ...mockPipeline, device: 'xpu' });

        expect(await screen.findByRole('button', { name: /inference device/i })).toHaveTextContent('GPU');
    });

    it('updates device on selection change', async () => {
        const pipelinePatchSpy = renderApp();

        const picker = await screen.findByRole('button', { name: /inference device/i });
        expect(picker).toHaveTextContent('CPU');

        await userEvent.click(picker);

        const option = await screen.findByRole('option', { name: /GPU/i });
        await userEvent.click(option);

        await waitFor(() => {
            expect(picker).toHaveTextContent('GPU');
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });
    });

    it('shows error toast on update failure', async () => {
        renderApp(500);

        await userEvent.click(await screen.findByRole('button', { name: /inference device/i }));
        await userEvent.click(screen.getByRole('option', { name: /GPU/i }));

        expect(await screen.findByLabelText('toast')).toBeVisible();
    });

    it('reverts selection on error', async () => {
        renderApp(500, mockDevices, { ...mockPipeline, device: 'cpu' });

        const picker = await screen.findByRole('button', { name: /inference device/i });
        expect(picker).toHaveTextContent('CPU');

        await userEvent.click(picker);
        await userEvent.click(screen.getByRole('option', { name: /GPU/i }));

        await screen.findByLabelText('toast');

        expect(picker).toHaveTextContent('CPU');
    });

    it('renders devices with same type and name correctly', async () => {
        const devicesResponse = [
            { type: 'cpu' as const, name: 'CPU', memory: null, index: null },
            { type: 'xpu' as const, name: 'Intel(R) Graphics [0xe216]', memory: 21359386624, index: 0 },
            { type: 'xpu' as const, name: 'Intel(R) Graphics [0xe216]', memory: 21359386624, index: 1 },
        ];
        const pipelinePatchSpy = renderApp(200, devicesResponse, { ...mockPipeline, device: 'cpu' });

        await userEvent.click(await screen.findByRole('button', { name: /inference device/i }));

        const options = await screen.findAllByRole('option');
        expect(options).toHaveLength(3);
        expect(options[0]).toHaveTextContent('CPU');
        expect(options[1]).toHaveTextContent('Intel(R) Graphics [0xe216]');
        expect(options[2]).toHaveTextContent('Intel(R) Graphics [0xe216]');

        // Pick the first xpu (index 0) and verify the patch uses 'xpu-0' (guards against falsy-check regressions)
        await userEvent.click(options[1]);

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalledWith({ device: 'xpu-0' });
        });
    });
});
