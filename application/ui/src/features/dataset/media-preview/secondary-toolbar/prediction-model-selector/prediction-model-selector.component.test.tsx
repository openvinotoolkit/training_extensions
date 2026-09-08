// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { PredictionsSetupProvider } from '../../../../annotator/predictions-setup-provider.component';
import { PredictionModelSelector } from './prediction-model-selector.component';

const STORAGE_KEY = '123-model-variant-id';

const modelA = getMockedModel({
    id: 'model-a',
    name: 'Model A',
    variants: [getMockedVariant({ id: 'variant-a', format: 'openvino', precision: 'fp16' })],
});

const modelB = getMockedModel({
    id: 'model-b',
    name: 'Model B',
    variants: [getMockedVariant({ id: 'variant-b', format: 'openvino', precision: 'fp32' })],
});

const getPicker = () => screen.getByRole('button', { name: /Select prediction model/ });

const renderApp = async (models: Model[]) => {
    server.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json(models)),
        http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(getMockedPipeline()))
    );

    render(
        <PredictionsSetupProvider>
            <PredictionModelSelector isDisabled={false} />
        </PredictionsSetupProvider>
    );

    await waitFor(() => {
        expect(getPicker()).toBeInTheDocument();
    });
};

describe('PredictionModelSelector', () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it('selects the only available model', async () => {
        await renderApp([modelA]);

        expect(getPicker()).toHaveTextContent('Model A [FP16]');
    });

    it('selects the only available model even if a different one was stored', async () => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify('variant-b'));

        await renderApp([modelA]);

        expect(getPicker()).toHaveTextContent('Model A [FP16]');
    });

    it('keeps the stored model when there is more than one to choose from', async () => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify('variant-b'));

        await renderApp([modelA, modelB]);

        expect(getPicker()).toHaveTextContent('Model B [FP32]');
    });
});
