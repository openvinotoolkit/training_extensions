// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { LocalFolderSinkConfig } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { useConnectSinkToPipeline } from '../../../../hooks/api/pipeline.hook';
import { server } from '../../../../msw-node-setup';
import { useSinkMutation } from '../hooks/use-sink-mutation.hook';
import { LocalFolder } from '../local-folder/local-folder.component';
import { getLocalFolderInitialConfig, localFolderBodyFormatter } from '../local-folder/utils';
import { OutputFormat } from '../utils';
import { EditSink } from './edit-sink.component';

rs.mock('../hooks/use-sink-mutation.hook');
rs.mock('../../../../hooks/api/pipeline.hook');

describe('EditSink', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        server.use(http.post('/api/sinks/{sink_id}:test', () => HttpResponse.json({ reachable: true })));
    });

    const updatedConfig: LocalFolderSinkConfig = {
        id: 'existing-sink-id',
        name: 'Updated folder sink',
        sink_type: 'folder',
        rate_limit: null,
        folder_path: '/data/output',
        output_formats: [],
    };

    const renderApp = (mockOnSaved = rs.fn(), isConnected = false) => {
        render(
            <EditSink
                config={{ ...getLocalFolderInitialConfig(), output_formats: [OutputFormat.PREDICTIONS] }}
                onSaved={mockOnSaved}
                onBackToList={rs.fn()}
                componentFields={(state: LocalFolderSinkConfig) => <LocalFolder defaultState={state} />}
                bodyFormatter={localFolderBodyFormatter}
                isConnected={isConnected}
            />
        );
    };

    it('calls connectToPipelineMutation after successful "Save & Connect" submit', async () => {
        const mockOnSaved = rs.fn();
        const mockSinkMutation = rs.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = rs.fn().mockResolvedValue(undefined);

        rs.mocked(useConnectSinkToPipeline).mockReturnValue(mockConnectToPipeline);
        rs.mocked(useSinkMutation).mockReturnValue(mockSinkMutation);

        renderApp(mockOnSaved);

        const folderPathInput = screen.getByRole('textbox', { name: /Folder Path/i });
        await userEvent.clear(folderPathInput);
        await userEvent.type(folderPathInput, updatedConfig.folder_path);
        await userEvent.click(screen.getByRole('button', { name: /Save & Connect/i }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSinkMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).toHaveBeenCalledWith(updatedConfig.id);
        });
    });

    it('does not call connectToPipelineMutation after successful "Save" submit', async () => {
        const mockOnSaved = rs.fn();
        const mockSinkMutation = rs.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = rs.fn().mockResolvedValue(undefined);

        rs.mocked(useConnectSinkToPipeline).mockReturnValue(mockConnectToPipeline);
        rs.mocked(useSinkMutation).mockReturnValue(mockSinkMutation);

        renderApp(mockOnSaved);

        const folderPathInput = screen.getByRole('textbox', { name: /Folder Path/i });
        await userEvent.clear(folderPathInput);
        await userEvent.type(folderPathInput, updatedConfig.folder_path);
        await userEvent.click(screen.getByRole('button', { name: 'Save' }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSinkMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).not.toHaveBeenCalled();
        });
    });

    it('hides "Save & Connect" button when already connected', () => {
        rs.mocked(useConnectSinkToPipeline).mockReturnValue(rs.fn());
        rs.mocked(useSinkMutation).mockReturnValue(rs.fn());

        renderApp(rs.fn(), true);

        expect(screen.queryByRole('button', { name: /Save & Connect/i })).not.toBeInTheDocument();
    });

    it('shows "Save & Connect" button when not connected', () => {
        rs.mocked(useConnectSinkToPipeline).mockReturnValue(rs.fn());
        rs.mocked(useSinkMutation).mockReturnValue(rs.fn());

        renderApp(rs.fn(), false);

        expect(screen.getByRole('button', { name: /Save & Connect/i })).toBeInTheDocument();
    });
});
