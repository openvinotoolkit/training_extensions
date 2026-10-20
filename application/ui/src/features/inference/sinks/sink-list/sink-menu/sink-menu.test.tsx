// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { SinkMenu, SinkMenuProps } from './sink-menu.component';

describe('SinkMenu', () => {
    const renderApp = ({
        id = 'id-test',
        name = 'name test',
        isConnected = false,
        onEdit = rs.fn(),
        onTest = rs.fn(),
    }: Partial<SinkMenuProps>) => {
        render(<SinkMenu id={id} name={name} isConnected={isConnected} onEdit={onEdit} onTest={onTest} />);
    };

    it('test connection', async () => {
        const onTest = rs.fn().mockResolvedValue(true);
        renderApp({ onTest });

        await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Test connection/i }));

        expect(onTest).toHaveBeenCalled();
    });

    it('edit', async () => {
        const mockedOnEdit = rs.fn();

        renderApp({ onEdit: mockedOnEdit });

        await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Edit/i }));

        expect(mockedOnEdit).toHaveBeenCalled();
    });

    describe('remove', () => {
        const name = 'test-name';
        const configRequests = (status = 204) => {
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
                        { status: 200 }
                    );
                }),
                http.delete('/api/sinks/{sink_id}', () => HttpResponse.json(null, { status }))
            );

            return pipelinePatchSpy;
        };

        it('success', async () => {
            const pipelinePatchSpy = configRequests();

            renderApp({ name, isConnected: false });

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Remove/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(`${name} has been removed successfully!`);
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });

        it('disabled when sink is connected', async () => {
            renderApp({ name, isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));

            expect(screen.getByRole('menuitem', { name: /Remove/i })).toHaveAttribute('aria-disabled', 'true');
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

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /^Connect$/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(`Successfully connected to "${name}"`);
        });

        it('error', async () => {
            configRequests(500);

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /^Connect$/i }));

            await expect(await screen.findByLabelText('toast')).toHaveTextContent(
                'An unexpected error occurred. Please try again.'
            );
        });

        it('shows connect when sink is disconnected', async () => {
            renderApp({ name, isConnected: false });

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));

            expect(screen.getByRole('menuitem', { name: /^Connect$/i })).toBeVisible();
        });
    });

    describe('disconnect', () => {
        const name = 'test-name';

        it('successfully disconnects', async () => {
            const pipelinePatchSpy = rs.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                    const body = await request.json();

                    pipelinePatchSpy(body);

                    return HttpResponse.json(
                        {
                            project_id: '',
                            status: 'idle',
                            device: 'images_folder',
                        },
                        { status: 200 }
                    );
                })
            );

            renderApp({ name, isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Disconnect/i }));

            expect(await screen.findByLabelText('toast')).toHaveTextContent(`Successfully disconnected from "${name}"`);

            expect(pipelinePatchSpy).toHaveBeenCalledWith({
                sink_id: null,
            });
        });

        it('shows disconnect when sink is connected', async () => {
            renderApp({ name: 'test-name', isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /sink menu/i }));

            expect(screen.getByRole('menuitem', { name: /Disconnect/i })).toBeVisible();
        });
    });
});
