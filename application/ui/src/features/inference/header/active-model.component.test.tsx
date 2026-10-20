// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model, Pipeline } from '@/api/types';
import { fireEvent, screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ActiveModel } from './active-model.component';

const ovVariantA = getMockedVariant({ id: 'variant-a-fp16', format: 'openvino', precision: 'fp16' });
const ovVariantB = getMockedVariant({ id: 'variant-b-fp32', format: 'openvino', precision: 'fp32' });

const modelA = getMockedModel({ id: 'model-a', name: 'Model A', variants: [ovVariantA] });
const modelB = getMockedModel({ id: 'model-b', name: 'Model B', variants: [ovVariantB] });

const runningModel = getMockedModel({
    id: 'running-model',
    name: 'Running model',
    training_info: { ...getMockedModel().training_info, status: 'in_progress' },
    variants: [getMockedVariant({ id: 'variant-running', format: 'openvino', precision: 'fp16' })],
});
const failedModel = getMockedModel({
    id: 'model-failed',
    name: 'Failed Model',
    training_info: { ...getMockedModel().training_info, status: 'failed' },
    variants: [getMockedVariant({ id: 'variant-failed', format: 'openvino', precision: 'fp16' })],
});
const notStartedModel = getMockedModel({
    id: 'not-started-model',
    name: 'Not started model',
    training_info: { ...getMockedModel().training_info, status: 'not_started' },
    variants: [getMockedVariant({ id: 'variant-not-started', format: 'openvino', precision: 'fp16' })],
});
const modelWithDeletedFiles = getMockedModel({
    id: 'model-deleted-files',
    name: 'Model with deleted files',
    training_info: { ...getMockedModel().training_info, status: 'successful' },
    files_deleted: true,
    variants: [getMockedVariant({ id: 'variant-deleted', format: 'openvino', precision: 'fp16' })],
});

const modelWithOnnxVariant = getMockedModel({
    id: 'model-with-onnx',
    name: 'Model Onnx',
    variants: [getMockedVariant({ id: 'variant-onnx', format: 'onnx', precision: 'fp32' })],
});

describe('ActiveModel', () => {
    const renderApp = ({ models, pipeline }: { models: Model[]; pipeline: Pipeline }) => {
        const patchSpy = rs.fn();

        server.use(
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json(models)),
            http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(pipeline)),
            http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                patchSpy(await request.json());

                return HttpResponse.json({ project_id: '', status: 'idle', device: 'images_folder' });
            })
        );

        render(<ActiveModel />);

        return patchSpy;
    };

    beforeEach(() => {
        rs.resetAllMocks();
    });

    it('shows only openvino variants of successful models', async () => {
        renderApp({
            models: [modelA, modelB, runningModel, failedModel, notStartedModel, modelWithDeletedFiles],
            pipeline: getMockedPipeline(),
        });

        expect(await screen.findByLabelText('active model')).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));

        expect(await screen.findByRole('option', { name: 'Model A [FP16]' })).toBeVisible();
        expect(screen.getByRole('option', { name: 'Model B [FP32]' })).toBeVisible();
        expect(screen.queryByRole('option', { name: /Running model/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: /Not started model/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: /Failed Model/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: /Model with deleted files/i })).not.toBeInTheDocument();
    });

    it('does not show base model entries, only openvino variants', async () => {
        renderApp({ models: [modelA], pipeline: getMockedPipeline() });

        expect(await screen.findByLabelText('active model')).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));

        expect(await screen.findByRole('option', { name: 'Model A [FP16]' })).toBeVisible();
        expect(screen.queryByRole('option', { name: 'Model A' })).not.toBeInTheDocument();
    });

    it('does not render model picker when there are no models', async () => {
        renderApp({ models: [], pipeline: getMockedPipeline() });

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        await waitFor(() => {
            expect(screen.queryByRole('button', { name: /active model/i })).not.toBeInTheDocument();
        });
    });

    it('does not render model picker when models have no openvino variants', async () => {
        renderApp({ models: [modelWithOnnxVariant], pipeline: getMockedPipeline() });

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        await waitFor(() => {
            expect(screen.queryByRole('button', { name: /active model/i })).not.toBeInTheDocument();
        });
    });

    it('patches pipeline with both model_id and model_variant_id when a variant is selected', async () => {
        const patchSpy = renderApp({
            models: [modelA, modelB],
            pipeline: getMockedPipeline(),
        });

        await screen.findByLabelText('active model');

        fireEvent.click(screen.getByRole('button', { name: /active model/i }));
        fireEvent.click(await screen.findByRole('option', { name: 'Model B [FP32]' }));

        await waitFor(() => {
            expect(patchSpy).toHaveBeenCalledWith({ model_id: 'model-b', model_variant_id: 'variant-b-fp32' });
        });
    });
});
