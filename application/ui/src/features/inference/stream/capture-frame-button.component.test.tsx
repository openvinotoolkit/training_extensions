// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { CaptureFrameButton } from './capture-frame-button.component';
import { useWebRTCConnection } from './web-rtc-connection-provider';

vi.mock('./web-rtc-connection-provider');

describe('CaptureFrameButton', () => {
    const renderApp = ({
        pipelineStatus = 'running',
        sourceStatus = 'ok',
        streamStatus = 'connected',
    }: {
        pipelineStatus?: 'running' | 'idle';
        sourceStatus?: 'ok' | 'finished' | 'error';
        streamStatus?: 'connected' | 'idle';
    } = {}) => {
        const captureSpy = vi.fn();

        vi.mocked(useWebRTCConnection).mockReturnValue({
            status: streamStatus,
            start: vi.fn(),
            stop: vi.fn(),
            webRTCConnectionRef: { current: null },
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: pipelineStatus }));
            }),
            http.get('/api/projects/{project_id}/pipeline/health', () => {
                return HttpResponse.json({
                    status: pipelineStatus,
                    components: {
                        source: { status: sourceStatus, updated_at: '2026-09-01T06:55:28Z', message: null },
                        sink: { status: 'unavailable', updated_at: '2026-09-01T06:55:28Z', message: null },
                        model: { status: 'ok', updated_at: '2026-09-01T06:55:28Z', message: null },
                    },
                });
            }),
            http.post('/api/projects/{project_id}/pipeline:capture', () => {
                captureSpy();

                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<CaptureFrameButton />);

        return captureSpy;
    };

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('captures a frame while the stream is running', async () => {
        const captureSpy = renderApp();

        const button = await screen.findByRole('button', { name: 'Capture frame' });
        await waitFor(() => {
            expect(button).toBeEnabled();
        });

        await userEvent.click(button);

        await waitFor(() => {
            expect(captureSpy).toHaveBeenCalled();
        });
        expect(await screen.findByText('Frame captured and added to the dataset.')).toBeVisible();
    });

    it('is disabled when the stream is off', async () => {
        renderApp({ streamStatus: 'idle' });

        expect(await screen.findByRole('button', { name: 'Capture frame' })).toBeDisabled();
    });

    it('is disabled when the pipeline is not running', async () => {
        renderApp({ pipelineStatus: 'idle', streamStatus: 'idle' });

        expect(await screen.findByRole('button', { name: 'Capture frame' })).toBeDisabled();
    });

    it('is disabled when the source has no more frames to produce', async () => {
        renderApp({ sourceStatus: 'finished' });

        await waitFor(async () => {
            expect(await screen.findByRole('button', { name: 'Capture frame' })).toBeDisabled();
        });
    });
});
