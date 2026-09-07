// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitecture, TrainingRequestPayload } from '@/api/types';
import { NetworkFixture } from '@msw/playwright';
import { getMockedJob } from 'mocks/mock-job';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { HttpResponse } from 'msw';

import { TIMM_MODEL_ARCHITECTURE_ID } from '../../src/features/models/train-model/timm-model-configuration/utils';
import { expect, http, test } from '../fixtures';
import { MOCKED_TRAINING_CONFIGURATION } from './mocks';

const TIMM_CATALOG: Record<string, Record<string, string[]>> = {
    resnet: {
        resnet50: ['a1_in1k', 'gluon_in1k'],
        resnet101: ['a1h_in1k'],
    },
    efficientnet: {
        efficientnet_b0: ['ra_in1k'],
    },
};

const mockedModelArchitectures = [
    getMockedModelArchitecture({ id: 'Object_Detection_SSD', name: 'Object_Detection_SSD' }),
    getMockedModelArchitecture({
        id: TIMM_MODEL_ARCHITECTURE_ID,
        name: 'Other models (TIMM)',
        description: 'PyTorch Image Models (TIMM) is a large collection of SOTA image classification models.',
        license: 'varies by model',
        stats: null,
    }),
];

const getMockedTimmManifest = (family: string, variant: string, pretrainedTag: string): ModelArchitecture =>
    getMockedModelArchitecture({
        id: `image-classification-timm-${variant}.${pretrainedTag}`,
        name: `timm/${variant}.${pretrainedTag}`,
        task: 'classification',
        license: 'Apache-2.0',
        timm_metadata: { family, variant, pretrained_tag: pretrainedTag },
        stats: {
            gigaflops: 4.1,
            trainable_parameters: 25,
            benchmark_metrics: { imagenet_top1_accuracy: 80.4 },
        },
    });

const setupNetworkMocks = (network: NetworkFixture) => {
    const state: { submittedJobBody: TrainingRequestPayload | null } = { submittedJobBody: null };

    network.use(
        http.get('/api/projects/{project_id}/models', () => {
            return HttpResponse.json([
                getMockedModel({
                    id: 'model-rev-1',
                    name: 'ResNet50 Revision 1',
                    architecture: 'image-classification-timm-resnet50.a1_in1k',
                }),
            ]);
        }),
        http.get('/api/model_architectures', () => {
            return HttpResponse.json({ model_architectures: mockedModelArchitectures, top_picks: null });
        }),
        http.get('/api/model_architectures/timm/families', () => {
            return HttpResponse.json(Object.keys(TIMM_CATALOG));
        }),
        http.get('/api/model_architectures/timm/families/{family}/variants', ({ params }) => {
            return HttpResponse.json(Object.keys(TIMM_CATALOG[params.family] ?? {}));
        }),
        http.get('/api/model_architectures/timm/families/{family}/variants/{variant}/pretrained-tags', ({ params }) => {
            return HttpResponse.json(TIMM_CATALOG[params.family]?.[params.variant] ?? []);
        }),
        http.get('/api/model_architectures/timm/manifest', ({ request }) => {
            const query = new URL(request.url).searchParams;

            return HttpResponse.json(
                getMockedTimmManifest(
                    String(query.get('family')),
                    String(query.get('variant')),
                    String(query.get('pretrained_tag'))
                )
            );
        }),
        http.get('/api/projects/{project_id}/training_configuration', () => {
            return HttpResponse.json(MOCKED_TRAINING_CONFIGURATION);
        }),
        http.get('/api/projects/{project_id}/dataset/items', () => {
            return HttpResponse.json({
                items: [
                    { id: '1', subset: 'training', user_reviewed: false },
                    { id: '2', subset: 'training', user_reviewed: false },
                    { id: '3', subset: 'validation', user_reviewed: false },
                    { id: '4', subset: 'testing', user_reviewed: false },
                ],
                pagination: { total: 4, count: 4, limit: 10, offset: 0 },
            });
        }),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.post('/api/jobs', async ({ request }) => {
            state.submittedJobBody = (await request.json()) as TrainingRequestPayload;

            return HttpResponse.json(getMockedJob({ job_id: 'job-timm-1' }), { status: 201 });
        })
    );

    return state;
};

test.describe('TIMM model training flow', () => {
    test('trains a timm backbone resolved from the family, variant and pretrained tag', async ({
        modelsPage,
        network,
        page,
    }) => {
        const state = setupNetworkMocks(network);

        await modelsPage.goto();
        await modelsPage.openTrainModelDialog();

        const timmConfiguration = page.getByRole('heading', { name: 'TIMM model configuration' });

        await test.step('hides the timm configuration for non timm architectures', async () => {
            await modelsPage.selectModelArchitecture('Object_Detection_SSD');

            await expect(timmConfiguration).toBeHidden();
        });

        await test.step('shows the timm configuration once the timm architecture is selected', async () => {
            await modelsPage.selectModelArchitecture('Other models (TIMM)');

            await expect(timmConfiguration).toBeVisible();
        });

        await test.step('cannot start training before a backbone is resolved', async () => {
            await expect(page.getByRole('button', { name: 'Start' })).toBeDisabled();
        });

        await test.step('defaults the variant and pretrained tag once a family is selected', async () => {
            await modelsPage.selectPickerOption('Architecture family', 'resnet');

            await expect(page.getByLabel('Model variant', { exact: true })).toContainText('resnet50');
            await expect(page.getByLabel('Pretrained Weights', { exact: true })).toContainText('a1_in1k');
            await expect(page.getByText('License: Apache-2.0')).toBeVisible();
        });

        await test.step('offers the input weights trained on the resolved backbone', async () => {
            await modelsPage.selectPickerOption('Select input weights', 'ResNet50 Revision 1');

            await expect(page.getByLabel('Select input weights', { exact: true }).last()).toContainText(
                'ResNet50 Revision 1'
            );
        });

        await test.step('resets the pretrained tag and the input weights when the variant changes', async () => {
            await modelsPage.selectPickerOption('Model variant', 'resnet101');

            await expect(page.getByLabel('Pretrained Weights', { exact: true })).toContainText('a1h_in1k');
            await expect(page.getByLabel('Select input weights', { exact: true }).last()).toContainText(
                'Default pre-trained weights'
            );
        });

        await modelsPage.startTraining();

        await expect
            .poll(() => state.submittedJobBody)
            .toMatchObject({
                job_type: 'train',
                project_id: 'id-1',
                parameters: {
                    device: 'cpu',
                    model_architecture_id: 'image-classification-timm-resnet101.a1h_in1k',
                    parent_model_revision_id: null,
                },
            });
    });
});
