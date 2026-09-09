// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { FEATURE_FLAGS } from '../../src/constants/feature-flags';
import { expect, http, test } from '../fixtures';

test.describe('Inference', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'id-1' }));
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(getMockedPipeline({ status: 'idle' }));
            }),
            http.get('/api/sources', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/sinks', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/system/devices/camera', () => {
                return HttpResponse.json([
                    {
                        index: 1,
                        name: 'FaceTime HD Camera',
                    },
                ]);
            }),
            http.post('/api/sources', () => {
                return HttpResponse.json(
                    {
                        id: 'generated-source-id',
                        name: 'Default Source',
                        source_type: 'usb_camera',
                        device_id: 0,
                    },
                    { status: 201 }
                );
            }),
            http.post('/api/sinks', () => {
                return HttpResponse.json(
                    {
                        id: 'generated-sink-id',
                        name: 'Default Sink',
                        sink_type: 'folder',
                        rate_limit: 5,
                        folder_path: '/default/path',
                        output_formats: ['predictions'],
                    },
                    { status: 201 }
                );
            })
        );
    });

    test('Inference workflow', async ({ streamPage, inferencePage, page, network }) => {
        await test.step('starts stream', async () => {
            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ status: 'running' }));
                })
            );

            await page.goto('/projects/id-1/inference');

            await streamPage.startStream();

            expect(streamPage.isConnected()).toBeTruthy();
        });

        await test.step('toggles pipeline', async () => {
            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ status: 'idle' }));
                })
            );

            await page.goto('/projects/id-1/inference');

            await expect(inferencePage.getPipelineSwitch('disabled')).toBeEnabled();

            network.use(
                http.post('/api/projects/{project_id}/pipeline:enable', () => {
                    return HttpResponse.json(null, { status: 204 });
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ status: 'running' }));
                })
            );

            await inferencePage.enablePipeline();

            await expect(inferencePage.getPipelineSwitch('enabled')).toBeEnabled();
            network.use(
                http.post('/api/projects/{project_id}/pipeline:disable', () => {
                    return HttpResponse.json(null, { status: 204 });
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ status: 'idle' }));
                })
            );

            await inferencePage.disablePipeline();

            await expect(inferencePage.getPipelineSwitch('disabled')).toBeEnabled();
        });

        await test.step('updates data collection policy', async () => {
            await page.goto('/projects/id-1/inference');

            // Open both tabs just to make sure everything works
            await page.getByRole('button', { name: 'Toggle Pipeline metrics tab' }).click();
            await expect(page.getByText('Pipeline metrics', { exact: true })).toBeVisible();

            await page.getByRole('button', { name: 'Toggle Data collection policy' }).click();
            await expect(page.getByRole('heading', { name: 'Data collection' })).toBeVisible();

            await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).not.toBeChecked();

            network.use(
                http.patch('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json({
                        project_id: '',
                        status: 'idle',
                        device: 'images_folder',
                    });
                }),
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 12,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: false,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 700,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 12,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: false,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            const maxDatasetSizeField = page.getByRole('textbox', { name: 'Size' });

            await maxDatasetSizeField.fill('700');
            await expect(maxDatasetSizeField).toHaveValue('700');

            await page.getByRole('switch', { name: 'Toggle auto capturing' }).click();
            await expect(page.getByRole('switch', { name: 'Toggle auto capturing' })).toBeChecked();

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 20,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: false,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            const framesField = page.getByRole('textbox', { name: 'Frames' });
            const secondsField = page.getByRole('textbox', { name: 'Seconds' });

            await expect(framesField).toBeEnabled();
            await expect(secondsField).toBeEnabled();

            await framesField.fill('20');
            await expect(framesField).toHaveValue('20');

            await expect(page.getByRole('switch', { name: 'Confidence threshold' })).not.toBeChecked();

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 20,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: true,
                                        confidence_threshold: 0.5,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            await page.getByRole('switch', { name: 'Confidence threshold' }).click();
            await expect(page.getByRole('switch', { name: 'Confidence threshold' })).toBeChecked();

            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(
                        getMockedPipeline({
                            data_collection: {
                                max_dataset_size: 500,
                                policies: [
                                    {
                                        type: 'fixed_rate',
                                        enabled: true,
                                        rate: 20,
                                    },
                                    {
                                        type: 'confidence_threshold',
                                        enabled: true,
                                        confidence_threshold: 0.7,
                                        min_sampling_interval: 2.5,
                                    },
                                ],
                            },
                        })
                    );
                })
            );

            const confidenceSlider = page.getByRole('slider', { name: 'Threshold' });
            await expect(confidenceSlider).toBeVisible();
            await expect(confidenceSlider).toBeEnabled();
            await confidenceSlider.fill('0.7');
            await expect(confidenceSlider).toHaveValue('0.7');
        });

        // TODO: drop the guard once CONFIDENCE_THRESHOLD is enabled by default
        if (FEATURE_FLAGS.CONFIDENCE_THRESHOLD) {
            await test.step('updates the inference confidence threshold', async () => {
                let confidenceThreshold = 0.35;
                const patchedBodies: unknown[] = [];

                network.use(
                    http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                        return response(200).json(
                            getMockedPipeline({
                                status: 'idle',
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
                                inference: { confidence_threshold: confidenceThreshold },
                            })
                        );
                    }),
                    http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                        const body = (await request.json()) as { inference?: { confidence_threshold: number } };
                        patchedBodies.push(body);

                        if (body.inference !== undefined) {
                            confidenceThreshold = body.inference.confidence_threshold;
                        }

                        return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                    })
                );

                await page.goto('/projects/id-1/inference');

                const thresholdField = page.getByRole('textbox', { name: 'Change Confidence threshold' });

                await expect(thresholdField).toHaveValue('0.35');

                await thresholdField.fill('0.8');
                await thresholdField.press('Enter');

                await expect(thresholdField).toHaveValue('0.8');
                await expect.poll(() => patchedBodies).toContainEqual({ inference: { confidence_threshold: 0.8 } });
            });
        }

        await test.step('updates input and output source', async () => {
            network.use(
                http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                    return response(200).json(getMockedPipeline({ source: null, sink: null }));
                }),
                http.post('/api/sources', () => {
                    return HttpResponse.json(
                        {
                            id: 'generated-source-id',
                            name: 'My Source',
                            source_type: 'usb_camera',
                            device_id: 1,
                        },
                        { status: 201 }
                    );
                }),
                http.post('/api/sinks', () => {
                    return HttpResponse.json(
                        {
                            id: 'generated-sink-id',
                            name: 'My Sink',
                            sink_type: 'folder',
                            rate_limit: 5,
                            folder_path: 'e2e-output',
                            output_formats: ['predictions'],
                        },
                        { status: 201 }
                    );
                }),
                http.patch('/api/sources/{source_id}', () => {
                    return HttpResponse.json({});
                }),
                http.patch('/api/sinks/{sink_id}', () => {
                    return HttpResponse.json({});
                })
            );
            await page.goto('/projects/id-1/inference');

            const usbCamera = 'My Source';

            network.use(
                http.get('/api/sources', () => {
                    return HttpResponse.json([
                        {
                            id: '1',
                            name: usbCamera,
                            source_type: 'usb_camera',
                            device_id: 1,
                        },
                    ]);
                })
            );

            network.use(
                http.get('/api/sinks', () => {
                    return HttpResponse.json([
                        {
                            id: '1',
                            name: 'My Sink',
                            sink_type: 'folder',
                            folder_path: 'e2e-output',
                            rate_limit: 5,
                            output_formats: ['predictions'],
                        },
                    ]);
                })
            );

            await expect(inferencePage.getAddSourceButton()).toBeVisible();

            await inferencePage.addUsbCameraSource({ name: usbCamera });
            await inferencePage.addFolderSink({
                name: 'My Sink',
                folderPath: 'my-output',
                outputFormats: ['Predictions'],
                rateLimitSamples: 5,
            });

            await inferencePage.getInputTab().click();
            await expect(inferencePage.getSourceCard(usbCamera)).toBeVisible();
            await expect(page.getByText('Device: FaceTime HD Camera')).toBeVisible();

            await inferencePage.getOutputTab().click();
            await expect(inferencePage.getSinkCard('My Sink')).toBeVisible();
            await expect(page.getByText('Folder path: e2e-output')).toBeVisible();
            await expect(page.getByText('Rate limit: 5 samples every 1 second')).toBeVisible();
            await expect(page.getByText('Output formats: predictions')).toBeVisible();
        });
    });

    test('shows stream only for projects with enabled pipeline', async ({
        page,
        network,
        streamPage,
        inferencePage,
    }) => {
        const projectWithEnabledPipeline = getMockedProject({
            id: 'enabled-project-id',
            name: 'Enabled project',
            active_pipeline: true,
        });

        const projectWithDisabledPipeline = getMockedProject({
            id: 'disabled-project-id',
            name: 'Disabled project',
            active_pipeline: false,
        });

        network.use(
            http.get('/api/projects', ({ response }) => {
                return response(200).json([projectWithEnabledPipeline, projectWithDisabledPipeline]);
            }),
            http.get('/api/projects/{project_id}', ({ params }) => {
                return HttpResponse.json(
                    params.project_id === projectWithEnabledPipeline.id
                        ? projectWithEnabledPipeline
                        : projectWithDisabledPipeline
                );
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ params, response }) => {
                return response(200).json(
                    getMockedPipeline({
                        project_id: params.project_id,
                        status: params.project_id === projectWithEnabledPipeline.id ? 'running' : 'idle',
                    })
                );
            })
        );

        await page.goto(`/projects/${projectWithEnabledPipeline.id}/inference`);
        await expect(streamPage.getStartStreamButton()).toBeVisible();

        await streamPage.startStream();
        await expect(streamPage.getStartStreamButton()).toBeHidden();

        await page.getByRole('button', { name: `Selected project ${projectWithEnabledPipeline.name}` }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        await page.getByRole('listitem').filter({ hasText: projectWithDisabledPipeline.name }).click();

        await expect(page).toHaveURL(new RegExp(`/projects/${projectWithDisabledPipeline.id}/dataset$`));

        await page.keyboard.press('Escape');
        await expect(page.getByRole('dialog')).toBeHidden();

        await inferencePage.openInferenceTab();

        await expect(page.getByTitle('Enable pipeline to start stream')).toBeVisible();
        await expect(page.getByRole('switch', { name: /Pipeline disabled/i })).toBeVisible();
    });

    test('resets stream state when re-opening inference page', async ({ streamPage, page, network }) => {
        network.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json(getMockedPipeline({ project_id: 'id-1', status: 'running' }));
            })
        );

        await page.goto('/projects/id-1/inference');

        await expect(streamPage.getStartStreamButton()).toBeVisible();
        await streamPage.startStream();

        await expect(streamPage.getStartStreamButton()).toBeHidden();

        await page.getByRole('tab', { name: 'Models' }).click();

        await expect(page).toHaveURL(new RegExp(`/projects/id-1/models$`));
        await page.getByRole('tab', { name: 'Inference' }).click();

        await expect(streamPage.getStartStreamButton()).toBeVisible();
    });
});
