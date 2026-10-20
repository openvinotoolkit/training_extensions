// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { IPCameraSourceConfig } from '@/api/types';
import { createI18nInstance } from '@/i18n';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { useConnectSourceToPipeline } from '../../../../hooks/api/pipeline.hook';
import { server } from '../../../../msw-node-setup';
import { useSourceMutation } from '../hooks/use-source-mutation.hook';
import { IpCamera } from '../ip-camera/ip-camera.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from '../ip-camera/utils';
import { EditSource } from './edit-source.component';

rs.mock('../hooks/use-source-mutation.hook');
rs.mock('../../../../hooks/api/pipeline.hook');

describe('EditIpCamera', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    beforeEach(() => {
        rs.clearAllMocks();
        server.use(http.post('/api/sources/{source_id}:test', () => HttpResponse.json({ reachable: true })));
    });

    const updatedConfig: IPCameraSourceConfig = {
        id: 'existing-source-id',
        name: 'Updated Camera',
        stream_url: 'rtsp://192.168.1.201:554/stream',
        source_type: 'ip_camera',
        auth_required: true,
    };

    const renderApp = (mockOnSaved = rs.fn(), isConnected = false) => {
        render(
            <EditSource
                config={getIpCameraInitialConfig(t)}
                onSaved={mockOnSaved}
                onBackToList={rs.fn()}
                componentFields={(state: IPCameraSourceConfig) => <IpCamera defaultState={state} />}
                bodyFormatter={ipCameraBodyFormatter}
                isConnected={isConnected}
            />
        );
    };

    it('calls connectToPipelineMutation after successful "Save & Connect" submit', async () => {
        const mockOnSaved = rs.fn();
        const mockSourceMutation = rs.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = rs.fn().mockResolvedValue(undefined);

        rs.mocked(useConnectSourceToPipeline).mockReturnValue(mockConnectToPipeline);
        rs.mocked(useSourceMutation).mockReturnValue(mockSourceMutation);

        renderApp(mockOnSaved);

        const nameInput = screen.getByRole('textbox', { name: /Name/i });
        const streamUrlInput = screen.getByRole('textbox', { name: /Stream Url/i });

        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, updatedConfig.name);
        await userEvent.clear(streamUrlInput);
        await userEvent.type(streamUrlInput, updatedConfig.stream_url);
        await userEvent.click(screen.getByRole('button', { name: /Save & Connect/i }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSourceMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).toHaveBeenCalledWith(updatedConfig.id);
        });
    });

    it('does not call connectToPipelineMutation after successful "Save" submit', async () => {
        const mockOnSaved = rs.fn();
        const mockSourceMutation = rs.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = rs.fn().mockResolvedValue(undefined);

        rs.mocked(useConnectSourceToPipeline).mockReturnValue(mockConnectToPipeline);
        rs.mocked(useSourceMutation).mockReturnValue(mockSourceMutation);

        renderApp(mockOnSaved);

        const nameInput = screen.getByRole('textbox', { name: /Name/i });
        const streamUrlInput = screen.getByRole('textbox', { name: /Stream Url/i });

        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, updatedConfig.name);
        await userEvent.clear(streamUrlInput);
        await userEvent.type(streamUrlInput, updatedConfig.stream_url);
        await userEvent.click(screen.getByRole('button', { name: /^Save$/i }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSourceMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).not.toHaveBeenCalled();
        });
    });

    it('hides "Save & Connect" button when already connected', () => {
        rs.mocked(useConnectSourceToPipeline).mockReturnValue(rs.fn());
        rs.mocked(useSourceMutation).mockReturnValue(rs.fn());

        renderApp(rs.fn(), true);

        expect(screen.queryByRole('button', { name: /Save & Connect/i })).not.toBeInTheDocument();
    });

    it('shows "Save & Connect" button when not connected', () => {
        rs.mocked(useConnectSourceToPipeline).mockReturnValue(rs.fn());
        rs.mocked(useSourceMutation).mockReturnValue(rs.fn());

        renderApp(rs.fn(), false);

        expect(screen.getByRole('button', { name: /Save & Connect/i })).toBeInTheDocument();
    });
});
