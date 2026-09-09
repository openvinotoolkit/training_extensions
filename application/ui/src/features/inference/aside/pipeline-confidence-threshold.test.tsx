// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Pipeline } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { PipelineConfidenceThreshold } from './pipeline-confidence-threshold.component';

const getInput = () => screen.getByRole('textbox', { name: 'Change Confidence threshold' });

const renderApp = (pipeline: Pipeline) => {
    const pipelinePatchSpy = vi.fn();

    server.use(
        http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(pipeline)),
        http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
            pipelinePatchSpy(await request.json());

            return HttpResponse.json(pipeline);
        })
    );

    render(<PipelineConfidenceThreshold />);

    return pipelinePatchSpy;
};

describe('PipelineConfidenceThreshold', () => {
    const mockedPipeline = getMockedPipeline({
        model_variant: {
            id: 'variant-id',
            model_revision_id: 'model-id',
            format: 'openvino',
            precision: 'fp16',
            weights_size: 1024,
            evaluations: [],
            files_deleted: false,
            optimal_confidence_threshold: 0.65,
        },
        inference: { confidence_threshold: 0.35 },
    });

    it("shows the pipeline's confidence threshold", async () => {
        renderApp(mockedPipeline);

        expect(await screen.findByRole('textbox', { name: 'Change Confidence threshold' })).toHaveValue('0.35');
    });

    it('is not rendered when the pipeline has no confidence threshold', async () => {
        renderApp(getMockedPipeline({ model_variant: null, inference: { confidence_threshold: null } }));

        await waitFor(() => {
            expect(screen.queryByRole('textbox', { name: 'Change Confidence threshold' })).not.toBeInTheDocument();
        });
    });

    it('persists a committed value through the pipeline', async () => {
        const pipelinePatchSpy = renderApp(mockedPipeline);

        await screen.findByRole('textbox', { name: 'Change Confidence threshold' });

        await userEvent.clear(getInput());
        await userEvent.type(getInput(), '0.8');
        await userEvent.tab();

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalledWith({ inference: { confidence_threshold: 0.8 } });
        });
    });

    it('resets to the optimal threshold of the pipeline model variant', async () => {
        const pipelinePatchSpy = renderApp(mockedPipeline);

        await screen.findByRole('textbox', { name: 'Change Confidence threshold' });
        await userEvent.click(screen.getByRole('button', { name: 'Reset confidence threshold' }));

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalledWith({ inference: { confidence_threshold: 0.65 } });
        });
    });

    it('restores the pipeline value when it could not be persisted', async () => {
        renderApp(mockedPipeline);

        server.use(http.patch('/api/projects/{project_id}/pipeline', () => HttpResponse.json(null, { status: 500 })));

        await screen.findByRole('textbox', { name: 'Change Confidence threshold' });

        expect(getInput()).toHaveValue('0.35');

        await userEvent.clear(getInput());
        await userEvent.type(getInput(), '0.8');
        await userEvent.tab();

        await waitFor(() => {
            expect(getInput()).toHaveValue('0.35');
        });
    });
});
