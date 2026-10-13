// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { defineNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';
import { getMockedDatasetStatistics } from 'mocks/mock-dataset-item';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { tusUploadHandlers } from 'mocks/mock-tus-upload';
import { HttpResponse } from 'msw';

import { handlers, http } from '../src/api/utils';
import { BoundingBoxToolPage } from './annotator/bounding-box-tool-page';
import { PolygonToolPage } from './annotator/polygon-tool-page';
import { SSIMToolPage } from './annotator/ssim-tool-page';
import { VideoPage } from './annotator/video/video-page';
import { AnnotatorPage } from './datasets/annotator-page';
import { DatasetPage } from './datasets/dataset-page';
import { ImportDatasetPage } from './datasets/import-dataset-page';
import { InferencePage } from './inference/inference-page';
import { StreamPage } from './inference/stream-page';
import { JobsDialogPage } from './jobs/jobs-dialog-page';
import { JobsPage } from './jobs/jobs-page';
import { ModelsPage } from './models/models-page';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
type DefineNetworkFixtureOptions = Parameters<typeof defineNetworkFixture>[0];
const sampleImagePath = path.resolve(dirname, './assets/candy-thumbnail.png');

let cachedSampleImage: ArrayBuffer | undefined;

const getSampleImageArrayBuffer = (): ArrayBuffer => {
    if (cachedSampleImage === undefined) {
        const sampleImageBuffer = fs.readFileSync(sampleImagePath);

        cachedSampleImage = sampleImageBuffer.buffer.slice(
            sampleImageBuffer.byteOffset,
            sampleImageBuffer.byteOffset + sampleImageBuffer.byteLength
        );
    }

    return cachedSampleImage;
};

interface Fixtures {
    network: NetworkFixture;
    streamPage: StreamPage;
    inferencePage: InferencePage;
    modelsPage: ModelsPage;
    jobsPage: JobsPage;
    jobsDialogPage: JobsDialogPage;
    polygonTool: PolygonToolPage;
    boundingBoxTool: BoundingBoxToolPage;
    ssimTool: SSIMToolPage;
    videoPage: VideoPage;
    annotatorPage: AnnotatorPage;
    importDatasetPage: ImportDatasetPage;
    datasetPage: DatasetPage;
}

const test = testBase.extend<Fixtures>({
    network: [
        async ({ context }, use) => {
            const network = defineNetworkFixture({
                context: context as DefineNetworkFixtureOptions['context'],
                // The stateful TUS handlers go first so they take precedence over the example
                // responses generated from the OpenAPI spec (mirrors `src/msw-node-setup.ts`).
                handlers: [
                    ...tusUploadHandlers,
                    ...handlers,
                    http.get('/health', ({ response }) => {
                        return response(200).json({
                            status: 'ok',
                        });
                    }),
                    http.get('/api/system/info', ({ response }) => {
                        return response(200).json({
                            license_accepted: true,
                            platform: 'linux',
                        });
                    }),
                    http.get('/api/system/metrics/memory', ({ response }) => {
                        return response(200).json({});
                    }),
                    http.get('/api/projects/{project_id}/models', ({ response }) => {
                        return response(200).json([]);
                    }),
                    http.get('/api/model_architectures', () => {
                        const mockedModelArchitectures = [
                            getMockedModelArchitecture({ id: 'Object_Detection_SSD', name: 'Object_Detection_SSD' }),
                            getMockedModelArchitecture({
                                id: 'Object_Detection_YOLOX_X',
                                name: 'Object_Detection_YOLOX_X',
                            }),
                            getMockedModelArchitecture({
                                id: 'Custom_Object_Detection_Gen3_ATSS',
                                name: 'Custom_Object_Detection_Gen3_ATSS',
                            }),
                        ];

                        return HttpResponse.json({
                            model_architectures: mockedModelArchitectures,
                            top_picks: {
                                balance: mockedModelArchitectures[0].id,
                                speed: mockedModelArchitectures[1].id,
                                accuracy: mockedModelArchitectures[2].id,
                            },
                        });
                    }),
                    http.get('/api/system/devices/training', () => {
                        return HttpResponse.json([{ type: 'cpu', name: 'CPU' }]);
                    }),
                    http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                        return response(200).json({
                            project_id: 'id-1',
                            status: 'idle',
                            source: null,
                            sink: null,
                            model: null,
                            device: 'cpu',
                        });
                    }),
                    http.post('/api/projects/{project_id}/pipeline:enable', () => {
                        return HttpResponse.json(null, { status: 204 });
                    }),
                    http.post('/api/projects/{project_id}/pipeline:disable', () => {
                        return HttpResponse.json(null, { status: 204 });
                    }),
                    http.get('/api/projects', ({ response }) => {
                        return response(200).json([
                            {
                                id: 'id-1',
                                name: 'Project 1',
                                task: {
                                    task_type: 'detection',
                                    exclusive_labels: false,
                                    labels: [],
                                },
                                active_pipeline: false,
                                created_at: '2024-10-01T12:00:00Z',
                            },
                        ]);
                    }),
                    http.get('/api/projects/{project_id}', () => {
                        return HttpResponse.json({
                            id: '123',
                            name: 'Test Project',
                            task: {
                                task_type: 'detection',
                                exclusive_labels: false,
                                labels: [
                                    { id: '1', color: 'red', name: 'person' },
                                    { id: '2', color: 'blue', name: 'car' },
                                ],
                            },
                            active_pipeline: true,
                            created_at: '2024-10-01T12:00:00Z',
                        });
                    }),
                    http.delete('/api/projects/{project_id}', () => {
                        return HttpResponse.json(null, { status: 204 });
                    }),
                    http.post('/api/webrtc/offer', ({ response }) => {
                        return response(200).json({
                            type: 'answer',
                            sdp: 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n',
                        } as never);
                    }),
                    http.post('/api/webrtc/input_hook', ({ response }) => {
                        // Schema is empty, so we return an empty object
                        return response(200).json({} as never);
                    }),
                    http.get('/api/projects/{project_id}/dataset/media/{media_id}/thumbnail', () => {
                        return HttpResponse.arrayBuffer(getSampleImageArrayBuffer(), {
                            headers: { 'content-type': 'image/png' },
                        });
                    }),
                    http.get('/api/system/devices/inference', ({ response }) => {
                        return response(200).json([
                            { type: 'cpu', name: 'CPU' },
                            { type: 'xpu', name: 'XPU' },
                        ]);
                    }),
                    http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', ({ response }) => {
                        return response(200).json({ annotations: [], user_reviewed: false, subset: 'training' });
                    }),
                    http.get('/api/projects/{project_id}/dataset/media', () => {
                        return HttpResponse.json({
                            items: [getMockedMediaImage({ width: 1000, height: 750 })],
                            pagination: { offset: 0, limit: 20, count: 1, total: 1 },
                        });
                    }),
                    http.get('/api/projects/{project_id}/dataset/items', () => {
                        return HttpResponse.json({
                            items: [],
                            pagination: { offset: 0, limit: 20, count: 0, total: 0 },
                        });
                    }),
                    http.get('/api/jobs/{job_id}/status', () => {
                        // Just a valid SSE response with no data
                        return new HttpResponse(':ok\n\n', {
                            status: 200,
                            headers: {
                                'Content-Type': 'text/event-stream',
                                'Cache-Control': 'no-cache',
                            },
                        });
                    }),
                    http.get('/api/projects/{project_id}/dataset/statistics', () => {
                        return HttpResponse.json(getMockedDatasetStatistics({}));
                    }),
                    http.get('/api/sources', () => {
                        return HttpResponse.json([]);
                    }),
                    http.get('/api/sinks', () => {
                        return HttpResponse.json([]);
                    }),
                    http.get('/api/projects/{project_id}/pipeline/health', () => {
                        return HttpResponse.json({
                            status: 'running',
                            components: {
                                source: { status: 'ok', updated_at: '2026-01-01T00:00:00Z', message: null },
                                sink: { status: 'ok', updated_at: '2026-01-01T00:00:00Z', message: null },
                                model: { status: 'ok', updated_at: '2026-01-01T00:00:00Z', message: null },
                            },
                        });
                    }),
                    http.get('/api/projects/{project_id}/dataset/views', () => {
                        return HttpResponse.json([getMockedDatasetView()]);
                    }),
                ],
            });

            await network.enable();
            await use(network);
            await network.disable();
        },
        { auto: true },
    ],
    streamPage: async ({ page }, use) => {
        const streamPage = new StreamPage(page);

        await use(streamPage);
    },
    inferencePage: async ({ page }, use) => {
        const inferencePage = new InferencePage(page);

        await use(inferencePage);
    },
    modelsPage: async ({ page }, use) => {
        const modelsPage = new ModelsPage(page);

        await use(modelsPage);
    },
    jobsPage: async ({ page }, use) => {
        const jobsPage = new JobsPage(page);

        await use(jobsPage);
    },
    jobsDialogPage: async ({ page }, use) => {
        const jobsDialogPage = new JobsDialogPage(page);

        await use(jobsDialogPage);
    },
    boundingBoxTool: async ({ page }, use) => {
        const boundingBoxTool = new BoundingBoxToolPage(page);
        await use(boundingBoxTool);
    },
    ssimTool: async ({ page }, use) => {
        const ssimTool = new SSIMToolPage(page);
        await use(ssimTool);
    },
    polygonTool: async ({ page }, use) => {
        const polygonTool = new PolygonToolPage(page);
        await use(polygonTool);
    },
    videoPage: async ({ page }, use) => {
        const videoPage = new VideoPage(page);
        await use(videoPage);
    },
    annotatorPage: async ({ page }, use) => {
        const annotatorPage = new AnnotatorPage(page);
        await use(annotatorPage);
    },
    importDatasetPage: async ({ page }, use) => {
        const importDatasetPage = new ImportDatasetPage(page);
        await use(importDatasetPage);
    },
    datasetPage: async ({ page }, use) => {
        const datasetPage = new DatasetPage(page);
        await use(datasetPage);
    },
});

export { expect, http, test };
