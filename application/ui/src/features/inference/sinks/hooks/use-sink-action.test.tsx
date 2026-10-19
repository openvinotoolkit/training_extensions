// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { startTransition } from 'react';

import type { LocalFolderSinkConfig, SinkOutputFormats } from '@/api/types';
import { act, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { useSinkAction } from './use-sink-action.hook';

const mockedConfig: LocalFolderSinkConfig = {
    id: '1',
    name: 'Test Folder',
    output_formats: [],
    sink_type: 'folder',
    folder_path: '',
    rate_limit: null,
};

const bodyFormatter = (formData: FormData) => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    sink_type: 'folder' as const,
    rate_limit:
        formData.get('rate_limit_samples') && formData.get('rate_limit_seconds')
            ? Number(formData.get('rate_limit_samples')) / Number(formData.get('rate_limit_seconds'))
            : null,
    folder_path: String(formData.get('folder_path')),
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
});

const renderApp = async ({
    isNewSink = false,
    config = mockedConfig,
    newSink = () => HttpResponse.error(),
    updateSink = () => HttpResponse.error(),
}) => {
    server.use(
        http.post('/api/sinks', newSink),
        http.patch('/api/sinks/{sink_id}', updateSink),
        http.patch('/api/projects/{project_id}/pipeline', () =>
            HttpResponse.json({
                project_id: '',
                status: 'idle',
                device: 'images_folder',
            })
        )
    );

    const { result } = renderHook(() => useSinkAction({ config: mockedConfig, isNewSink, bodyFormatter }));
    const [_state, submitAction] = result.current;

    const formData = new FormData();
    formData.append('name', config.name);
    formData.append('sink_type', config.sink_type);

    if (config.rate_limit !== null && config.rate_limit !== undefined) {
        formData.append('rate_limit_samples', String(config.rate_limit));
        formData.append('rate_limit_seconds', '1');
    }

    formData.append('folder_path', config.folder_path);
    formData.append('output_formats', config.output_formats.join(','));

    config.id && formData.append('id', config.id);

    await act(async () => {
        startTransition(async () => submitAction(formData));
    });

    return result.current;
};

describe('useSinkAction', () => {
    beforeEach(() => {
        server.use(http.post('/api/sinks/{sink_id}:test', () => HttpResponse.json({ reachable: true })));
    });

    it('return initial config', () => {
        const { result } = renderHook(() => useSinkAction({ config: mockedConfig, isNewSink: true, bodyFormatter }));

        expect(result.current[0]).toEqual(mockedConfig);
    });

    describe('new configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';
            await renderApp({
                isNewSink: true,
                newSink: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save sink configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submit new image folder config and show success message', async () => {
            const mockedNewItemId = 'new-id-test';
            const [state] = await renderApp({
                isNewSink: true,
                newSink: () => HttpResponse.json({ ...mockedConfig, id: mockedNewItemId }),
            });

            await waitFor(() => {
                expect(screen.getByText('Sink configuration created successfully.')).toBeVisible();
                expect(state.id).toBe(mockedNewItemId);
            });
        });
    });

    describe('edit configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';

            await renderApp({
                isNewSink: false,
                updateSink: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save sink configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submits folder config and show success message', async () => {
            const newConfig = { ...mockedConfig, id: 'mockedResponseId', name: 'Updated Name' };

            const [state] = await renderApp({
                isNewSink: false,
                config: newConfig,
                updateSink: () => HttpResponse.json(newConfig),
            });

            await waitFor(() => {
                expect(screen.getByText('Sink configuration updated successfully.')).toBeVisible();
                expect(state.id).toBe(newConfig.id);
            });
        });
    });
});
