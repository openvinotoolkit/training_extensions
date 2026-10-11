// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetSubset } from '@/api/types';
import { expect, type Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';
import { withRelative } from '../utils/mouse';

export class AnnotatorPage {
    constructor(private readonly page: Page) {}

    getProcessingImage() {
        return this.page.getByText('Processing image, please wait...');
    }

    // Hovers an image point with the Segment Anything tool and accepts its preview.
    async annotateAt(x: number, y: number) {
        const relative = await withRelative(this.page);
        const point = relative(x, y);

        await this.page.mouse.move(point.x, point.y);
        await expect(this.page.getByLabel('Segment anything preview')).toBeVisible({ timeout: 15_000 });

        await this.page.mouse.down();
        await this.page.mouse.up();
    }

    getAnnotationsList() {
        return this.page.getByTestId('annotation-layer');
    }

    async getAnnotationsListItems(label: 'annotation rect' | 'prediction rect' | 'annotation polygon') {
        return this.getAnnotationsList()
            .getByLabel(label)
            .evaluateAll((nodes) => nodes.filter((node) => !node.closest('mask')));
    }

    getAnnotatorMode(mode: 'annotation' | 'prediction') {
        return this.page.getByTestId('annotator-modes-id').getByRole('button', { name: mode });
    }

    async openAnnotationMode() {
        await this.getAnnotatorMode('annotation').click();
    }

    async openPredictionMode() {
        await this.getAnnotatorMode('prediction').click();
    }

    getPredictionSettingsButton() {
        return this.page.getByRole('button', { name: 'Prediction settings' });
    }

    async openPredictionSettings() {
        await this.getPredictionSettingsButton().click();
    }

    getPrimaryToolbar() {
        return this.page.getByLabel('primary toolbar');
    }

    async editPrediction() {
        await this.page.getByRole('button', { name: 'Edit prediction' }).click();
    }

    async goto(projectId: string, datasetItemId: string) {
        await this.page.goto(paths.project.dataset.item.index({ projectId, datasetItemId }));
    }

    async selectSubset(subset: DatasetSubset) {
        await this.page.getByRole('button', { name: /Select subset/ }).click();
        await this.page.getByRole('option', { name: new RegExp(subset, 'i') }).click();
    }

    getSelectedSubset() {
        return this.page.getByTestId('selected-subset-badge');
    }

    async submit() {
        await this.page.getByRole('button', { name: 'Submit' }).click();
    }

    async submitAndWaitForSave() {
        const responsePromise = this.page.waitForResponse((response) => {
            const url = new URL(response.url());

            return response.request().method() === 'POST' && url.pathname.endsWith('/annotations');
        });

        await this.submit();

        return responsePromise;
    }

    async selectMediaItem(name: string) {
        const sidebarItems = this.page.getByRole('listbox', { name: 'sidebar-items' });
        await sidebarItems.getByRole('img', { name, exact: true }).click();
    }

    getSelectedMediaItem() {
        return this.page
            .getByRole('listbox', { name: 'sidebar-items' })
            .locator('[aria-selected="true"]')
            .locator('img');
    }

    async close() {
        await this.page.getByRole('dialog').getByRole('button', { name: 'Close' }).click();
    }

    getMediaCanvasLoading() {
        return this.page.getByRole('progressbar', { name: 'Media canvas loading' });
    }

    // The loading spinner only shows after a delay, so it can be hidden while the image is still loading.
    async waitForMediaLoaded(timeout?: number) {
        await expect(this.page.getByTestId('media-canvas')).toHaveAttribute('aria-busy', 'false', { timeout });
    }
}
