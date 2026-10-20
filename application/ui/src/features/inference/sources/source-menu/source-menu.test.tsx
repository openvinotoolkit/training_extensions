// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import * as actualWebRtc from '../../stream/web-rtc-connection-provider' with { rstest: 'importActual' };
import { SourceMenu, SourceMenuProps } from './source-menu.component';

const mockStopStream = rs.fn();

rs.mock('../../stream/web-rtc-connection-provider', () => ({
    ...actualWebRtc,
    useWebRTCConnection: () => ({
        stop: mockStopStream,
    }),
}));

describe('SourceMenu', () => {
    beforeEach(() => {
        mockStopStream.mockReset();
    });

    const renderApp = ({
        id = 'id-test',
        name = 'name test',
        isConnected = false,
        onEdit = rs.fn(),
        isPipelineRunning = false,
        onTest = rs.fn(),
    }: Partial<SourceMenuProps>) => {
        render(
            <SourceMenu
                id={id}
                name={name}
                isConnected={isConnected}
                onEdit={onEdit}
                isPipelineRunning={isPipelineRunning}
                onTest={onTest}
            />
        );
    };

    it('test connection', async () => {
        const onTest = rs.fn().mockResolvedValue(true);
        renderApp({ onTest });

        await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Test connection/i }));

        expect(onTest).toHaveBeenCalled();
    });

    it('edit', async () => {
        const mockedOnEdit = rs.fn();

        renderApp({ onEdit: mockedOnEdit });

        await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Edit/i }));

        expect(mockedOnEdit).toHaveBeenCalled();
    });

    describe('remove', () => {
        const name = 'test-name';
        const configRequests = (status = 200) => {
            const pipelinePatchSpy = rs.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', () => {
                    pipelinePatchSpy();
                    return HttpResponse.json(
                        {
                            project_id: '',
                            status: 'idle',
                            device: 'images_folder',
                        },
                        { status }
                    );
                }),
                http.delete('/api/sources/{source_id}', () => HttpResponse.json(null, { status: 204 }))
            );

            return pipelinePatchSpy;
        };

        it('success', async () => {
            const pipelinePatchSpy = configRequests();

            renderApp({ name, isConnected: false });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Remove/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(`${name} has been removed successfully!`);
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });

        it('disabled when source is connected', async () => {
            renderApp({ name, isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));

            expect(screen.getByRole('menuitem', { name: /Remove/i })).toHaveAttribute('aria-disabled', 'true');
            expect(screen.getByRole('menuitem', { name: /Test connection/i })).toHaveAttribute('aria-disabled', 'true');
        });
    });

    describe('connect', () => {
        const name = 'test-name';
        const configRequests = (status = 200) => {
            server.use(
                http.patch('/api/projects/{project_id}/pipeline', () =>
                    HttpResponse.json(
                        {
                            project_id: '',
                            status: 'idle',
                            device: 'images_folder',
                        },
                        { status }
                    )
                )
            );
        };

        it('success', async () => {
            configRequests();

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /^Connect$/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(`Successfully connected to "${name}"`);
        });

        it('error', async () => {
            configRequests(500);

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /^Connect$/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(
                'An unexpected error occurred. Please try again.'
            );
        });

        it('shows connected when is disconnected', async () => {
            renderApp({ name, isConnected: false });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            expect(screen.getByRole('menuitem', { name: /^Connect$/i })).toBeVisible();
        });
    });

    describe('disconnect', () => {
        const name = 'test-name';

        it('successfully disconnects source when pipeline is not running', async () => {
            const pipelinePatchSpy = rs.fn();
            const disablePipeline = rs.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                    const body = await request.json();

                    pipelinePatchSpy(body);

                    return HttpResponse.json({
                        project_id: '',
                        status: 'idle',
                        device: 'images_folder',
                    });
                }),
                http.post('/api/projects/{project_id}/pipeline:disable', async () => {
                    disablePipeline();

                    return HttpResponse.json(null, { status: 204 });
                })
            );

            renderApp({ name, isConnected: true, isPipelineRunning: false });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Disconnect/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(`Successfully disconnected from "${name}"`);

            expect(disablePipeline).not.toHaveBeenCalled();
            expect(pipelinePatchSpy).toHaveBeenCalledWith({ source_id: null });
        });

        it('successfully disconnects source when pipeline is running and stops stream', async () => {
            const pipelinePatchSpy = rs.fn();
            const disablePipeline = rs.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                    const body = await request.json();

                    pipelinePatchSpy(body);

                    return HttpResponse.json({
                        project_id: '',
                        status: 'idle',
                        device: 'images_folder',
                    });
                }),
                http.post('/api/projects/{project_id}/pipeline:disable', async () => {
                    disablePipeline();

                    return HttpResponse.json(null, { status: 204 });
                })
            );

            renderApp({ name, isConnected: true, isPipelineRunning: true });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Disconnect/i }));

            expect(screen.getByRole('heading', { name: `Disconnect ${name}` })).toBeVisible();

            await userEvent.click(screen.getByRole('button', { name: 'Disconnect' }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(
                `Successfully disabled pipeline and disconnected from "${name}"`
            );

            expect(disablePipeline).toHaveBeenCalled();
            expect(pipelinePatchSpy).toHaveBeenCalledWith({ source_id: null });
            expect(disablePipeline.mock.invocationCallOrder[0]).toBeLessThan(
                pipelinePatchSpy.mock.invocationCallOrder[0]
            );

            expect(mockStopStream).toHaveBeenCalled();
        });

        it('shows disconnect when is connected', async () => {
            renderApp({ name, isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            expect(screen.getByRole('menuitem', { name: /Disconnect/i })).toBeVisible();
        });
    });
});
