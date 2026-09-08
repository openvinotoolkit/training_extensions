// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Page } from '@playwright/test';

import { DatasetViewsPage } from './dataset-views-page';

const pluralizeItems = (count: number) => {
    const pluralRules = new Intl.PluralRules('en');

    return pluralRules.select(count) === 'one' ? 'item' : 'items';
};

export class DatasetPage {
    readonly views: DatasetViewsPage;

    constructor(private readonly page: Page) {
        this.views = new DatasetViewsPage(page);
    }

    goto(projectId = 'id-1', search = '') {
        return this.page.goto(`projects/${projectId}/dataset${search}`);
    }

    async openAnnotator() {
        await this.page.getByRole('button', { name: 'Annotate' }).click();
    }

    getMediaGrid() {
        return this.page.getByRole('listbox', { name: 'data-collection-grid' });
    }

    getMediaItemById(mediaId: string) {
        return this.getMediaGrid()
            .getByRole('option')
            .filter({
                has: this.page.getByRole('checkbox', {
                    name: `Selection state of media item ${mediaId}`,
                    exact: true,
                }),
            });
    }

    async selectMediaItem(mediaId: string) {
        // The checkbox only reflects the state, selection happens on the item itself,
        // where a plain click replaces the selection and a ctrl click adds to it
        await this.getMediaItemById(mediaId).click({ modifiers: ['Control'] });
    }

    getMediaItemByName(name: string) {
        return this.page.getByRole('img', { name, exact: true });
    }

    dblClickMediaItem(name: string) {
        return this.getMediaItemByName(name).dblclick();
    }

    getSelectAllCheckbox() {
        return this.page.getByLabel('select all');
    }

    selectAll() {
        return this.getSelectAllCheckbox().click();
    }

    getSelectedCountText(count: number) {
        return this.page.getByText(`${count} selected`);
    }

    getImagesCountText(count: number) {
        return this.page.getByText(`${count} media ${pluralizeItems(count)}`);
    }

    getUploadInput() {
        return this.page.getByTestId('upload-media-input');
    }

    getUploadButton() {
        return this.page.getByRole('button', { name: 'Upload media' });
    }

    uploadFiles(files: { name: string; mimeType: string; buffer: Buffer }[] | string[]) {
        return this.getUploadInput().setInputFiles(files);
    }

    getUploadProgressText(total: number) {
        return this.page.getByText(`Uploading ${total} ${pluralizeItems(total)}...`);
    }

    getUploadProgressDetailText(succeeded: number, failed = 0) {
        const parts = [succeeded > 0 ? `${succeeded} succeeded` : null, failed > 0 ? `${failed} failed` : null]
            .filter(Boolean)
            .join(', ');
        return this.page.getByText(`(${parts})`);
    }

    getUploadFinishedText(total: number) {
        return this.page.getByText(`Uploaded ${total} ${pluralizeItems(total)}`);
    }

    getShowDetailsButton() {
        return this.page.getByRole('button', { name: 'Show details' });
    }

    clickShowDetails() {
        // Sonner stacks/re-renders toasts during upload progress updates, which can briefly cause
        // the toast container to intercept clicks on the button. Forcing the click bypasses that race.
        // eslint-disable-next-line playwright/no-force-option
        return this.getShowDetailsButton().click({ force: true });
    }

    getUploadDetailsDialog() {
        return this.page
            .getByRole('dialog')
            .filter({ has: this.page.getByRole('heading', { name: 'Upload details' }) });
    }

    getUploadDetailsRows() {
        return this.getUploadDetailsDialog().getByRole('row');
    }

    closeUploadDetailsDialog() {
        return this.getUploadDetailsDialog().getByRole('button', { name: 'Close' }).click();
    }

    getAssignLabelButton() {
        return this.page.getByRole('button', { name: 'Assign label' });
    }

    clickAssignLabel() {
        return this.getAssignLabelButton().click();
    }

    getLabelAssignmentDialog() {
        return this.page.getByRole('dialog');
    }

    getLabelAssignmentHeading() {
        return this.page.getByRole('heading', { name: 'Label assignment' });
    }

    getLabelCheckbox(labelName: string) {
        return this.page.getByRole('checkbox', { name: `Select ${labelName}` });
    }

    selectLabel(labelName: string) {
        return this.getLabelCheckbox(labelName).click();
    }

    getContinueButton() {
        return this.page.getByRole('button', { name: 'Continue' });
    }

    clickContinue() {
        return this.getContinueButton().click();
    }

    getSkipButton() {
        return this.page.getByRole('button', { name: 'Skip' });
    }

    clickSkip() {
        return this.getSkipButton().click();
    }

    getBulkDialogAssignButton() {
        return this.getLabelAssignmentDialog().getByRole('button', { name: 'Assign' });
    }

    clickDialogAssign() {
        return this.getBulkDialogAssignButton().click();
    }
}
