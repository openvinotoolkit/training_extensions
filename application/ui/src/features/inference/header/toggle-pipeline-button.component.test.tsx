// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { useWebRTCConnection } from '../stream/web-rtc-connection-provider';
import { TogglePipelineButton } from './toggle-pipeline-button.component';

rs.mock('../stream/web-rtc-connection-provider');

const mockStopStream = rs.fn().mockResolvedValue(undefined);

describe('TogglePipelineButton', () => {
    beforeEach(() => {
        rs.mocked(useWebRTCConnection).mockReturnValue({
            status: 'idle',
            start: rs.fn(),
            stop: mockStopStream,
            webRTCConnectionRef: { current: null },
        });
    });

    afterEach(() => {
        rs.clearAllMocks();
    });

    it('enables pipeline when currently idle', async () => {
        rs.mocked(useWebRTCConnection).mockReturnValue({
            status: 'connected',
            start: rs.fn(),
            stop: mockStopStream,
            webRTCConnectionRef: { current: null },
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('switch', { name: /Pipeline disabled/i }));

        expect(await screen.findByText('Pipeline enabled successfully')).toBeVisible();
        expect(mockStopStream).not.toHaveBeenCalled();
    });

    it('disables pipeline when currently running', async () => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'running' }));
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('switch', { name: /Pipeline enabled/i }));

        expect(await screen.findByText('Pipeline disabled successfully')).toBeVisible();
    });

    it('stops stream when pipeline is disabled and stream is active', async () => {
        rs.mocked(useWebRTCConnection).mockReturnValue({
            status: 'connected',
            start: rs.fn(),
            stop: mockStopStream,
            webRTCConnectionRef: { current: null },
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'running' }));
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('switch', { name: /Pipeline enabled/i }));

        await waitFor(() => {
            expect(mockStopStream).toHaveBeenCalledTimes(1);
        });
    });

    it('does not stop stream when pipeline is disabled and stream is already idle', async () => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'running' }));
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('switch', { name: /Pipeline enabled/i }));

        await screen.findByText('Pipeline disabled successfully');
        expect(mockStopStream).not.toHaveBeenCalled();
    });

    it('shows explanation dialog when pipeline is not configured', async () => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(
                    getMockedPipeline({
                        status: 'idle',
                        model: null,
                        source: null,
                    })
                );
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('switch', { name: /Pipeline disabled/i }));

        expect(await screen.findByText('Cannot enable pipeline')).toBeVisible();
        expect(
            await screen.findByText('Make sure you selected a model and source before enabling the pipeline.')
        ).toBeVisible();
    });
});
