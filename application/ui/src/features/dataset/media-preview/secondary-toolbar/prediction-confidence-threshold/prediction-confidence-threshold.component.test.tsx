// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { PredictionsSetupProvider } from '../../../../annotator/predictions-setup-provider.component';
import { PredictionModelSelector } from '../prediction-model-selector/prediction-model-selector.component';
import { PredictionConfidenceThreshold } from './prediction-confidence-threshold.component';

const modelA = getMockedModel({
    id: 'model-a',
    name: 'Model A',
    variants: [
        getMockedVariant({
            id: 'variant-a',
            format: 'openvino',
            precision: 'fp16',
            optimal_confidence_threshold: 0.65,
        }),
    ],
});

const modelB = getMockedModel({
    id: 'model-b',
    name: 'Model B',
    variants: [
        getMockedVariant({
            id: 'variant-b',
            format: 'openvino',
            precision: 'fp32',
            optimal_confidence_threshold: 0.2,
        }),
    ],
});

const getInput = () => screen.getByRole('textbox', { name: 'Change Confidence threshold' });

const renderApp = async (models: Model[]) => {
    server.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json(models)),
        http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(getMockedPipeline()))
    );

    render(
        <PredictionsSetupProvider>
            <PredictionModelSelector isDisabled={false} />
            <PredictionConfidenceThreshold />
        </PredictionsSetupProvider>
    );

    await waitFor(() => {
        expect(screen.getByRole('button', { name: /Select prediction model/ })).toBeInTheDocument();
    });
};

const selectModel = async (name: string) => {
    await userEvent.click(screen.getByRole('button', { name: /Select prediction model/ }));
    await userEvent.click(await screen.findByRole('option', { name }));
};

describe('PredictionConfidenceThreshold', () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it("shows the selected model variant's optimal confidence threshold", async () => {
        await renderApp([modelA]);

        expect(getInput()).toHaveValue('0.65');
    });

    it('is not rendered when the selected model variant has no optimal confidence threshold', async () => {
        const modelWithoutThreshold = getMockedModel({
            id: 'model-c',
            name: 'Model C',
            variants: [getMockedVariant({ id: 'variant-c', optimal_confidence_threshold: null })],
        });

        await renderApp([modelWithoutThreshold]);

        expect(screen.queryByRole('textbox', { name: 'Change Confidence threshold' })).not.toBeInTheDocument();
    });

    it('resets to the optimal confidence threshold of the newly selected model', async () => {
        await renderApp([modelA, modelB]);

        await selectModel('Model B [FP32]');
        expect(getInput()).toHaveValue('0.2');

        await userEvent.clear(getInput());
        await userEvent.type(getInput(), '0.9');
        await userEvent.tab();
        expect(getInput()).toHaveValue('0.9');

        await selectModel('Model A [FP16]');
        expect(getInput()).toHaveValue('0.65');
    });
});
