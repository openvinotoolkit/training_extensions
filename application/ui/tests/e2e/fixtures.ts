// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, test as testBase } from '@playwright/test';

import { BoundingBoxToolPage } from '../annotator/bounding-box-tool-page';
import { AnnotatorPage } from '../datasets/annotator-page';
import { DatasetPage } from '../datasets/dataset-page';
import { ImportDatasetPage } from '../datasets/import-dataset-page';
import { InferencePage } from '../inference/inference-page';
import { StreamPage } from '../inference/stream-page';
import { ModelsPage } from '../models/models-page';
import { ProjectPage } from '../projects/project-page';

interface Fixtures {
    acceptLicense: void;
    projectPage: ProjectPage;
    datasetPage: DatasetPage;
    importDatasetPage: ImportDatasetPage;
    annotatorPage: AnnotatorPage;
    boundingBoxTool: BoundingBoxToolPage;
    modelsPage: ModelsPage;
    inferencePage: InferencePage;
    streamPage: StreamPage;
}

const test = testBase.extend<Fixtures>({
    // A fresh backend would otherwise block every flow behind the license screen.
    acceptLicense: [
        async ({ request }, use) => {
            await request.post('/api/license/accept');
            await use();
        },
        { auto: true },
    ],
    projectPage: async ({ page }, use) => {
        const projectPage = new ProjectPage(page);

        await use(projectPage);
    },
    datasetPage: async ({ page }, use) => {
        const datasetPage = new DatasetPage(page);
        await use(datasetPage);
    },
    importDatasetPage: async ({ page }, use) => {
        const importDatasetPage = new ImportDatasetPage(page);
        await use(importDatasetPage);
    },
    annotatorPage: async ({ page }, use) => {
        const annotatorPage = new AnnotatorPage(page);
        await use(annotatorPage);
    },
    boundingBoxTool: async ({ page }, use) => {
        const boundingBoxTool = new BoundingBoxToolPage(page);
        await use(boundingBoxTool);
    },
    modelsPage: async ({ page }, use) => {
        const modelsPage = new ModelsPage(page);

        await use(modelsPage);
    },
    inferencePage: async ({ page }, use) => {
        const inferencePage = new InferencePage(page);

        await use(inferencePage);
    },
    streamPage: async ({ page }, use) => {
        const streamPage = new StreamPage(page);

        await use(streamPage);
    },
});

export { test, expect };
