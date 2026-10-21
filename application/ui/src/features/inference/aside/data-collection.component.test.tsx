// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { DataCollection } from './data-collection.component';

describe('DataCollection capture rate fields', () => {
    const renderApp = () => {
        const pipelinePatchSpy = rs.fn();

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(
                    getMockedPipeline({
                        data_collection: {
                            max_dataset_size: 500,
                            policies: [
                                {
                                    type: 'fixed_rate',
                                    rate: 12,
                                    enabled: true,
                                },
                                {
                                    type: 'confidence_threshold',
                                    confidence_threshold: 0.5,
                                    min_sampling_interval: 2.5,
                                    enabled: false,
                                },
                            ],
                        },
                    })
                );
            }),
            http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                pipelinePatchSpy(await request.json());

                return HttpResponse.json({
                    project_id: '',
                    status: 'idle',
                    device: 'images_folder',
                });
            })
        );

        render(<DataCollection />);

        return pipelinePatchSpy;
    };

    beforeEach(() => {
        rs.resetAllMocks();
    });

    it('renders Frames and Seconds fields for capture rate', async () => {
        renderApp();

        expect(await screen.findByLabelText('Frames')).toBeVisible();
        expect(screen.getByLabelText('Seconds')).toBeVisible();
        expect(screen.queryByLabelText('Rate')).not.toBeInTheDocument();
    });

    it('patches fixed_rate using frames divided by seconds on blur', async () => {
        const pipelinePatchSpy = renderApp();

        const framesInput = await screen.findByLabelText('Frames');
        const secondsInput = screen.getByLabelText('Seconds');

        await userEvent.clear(framesInput);
        await userEvent.type(framesInput, '6');
        framesInput.blur();

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalledWith(
                expect.objectContaining({
                    data_collection: expect.objectContaining({
                        policies: expect.arrayContaining([
                            expect.objectContaining({
                                type: 'fixed_rate',
                                rate: 6, // 6 frames / 1 second
                            }),
                        ]),
                    }),
                })
            );
        });

        await userEvent.clear(secondsInput);
        await userEvent.type(secondsInput, '3');
        secondsInput.blur();

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenLastCalledWith(
                expect.objectContaining({
                    data_collection: expect.objectContaining({
                        policies: expect.arrayContaining([
                            expect.objectContaining({
                                type: 'fixed_rate',
                                rate: 2, // 6 frames / 3 seconds
                            }),
                        ]),
                    }),
                })
            );
        });

        expect(pipelinePatchSpy).toHaveBeenCalledTimes(2);
    });
});
