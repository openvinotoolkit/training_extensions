// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Page } from '@playwright/test';

export class ImportDatasetPage {
    constructor(private readonly page: Page) {}

    getDialog() {
        return this.page.getByRole('dialog');
    }

    async openImportDialog() {
        await this.page.getByRole('button', { name: 'import export dataset' }).click();
        await this.page.getByText('Import dataset').click();
    }

    async loadZipFile(name: string) {
        const fileChooserPromise = this.page.waitForEvent('filechooser');
        await this.getDialog().getByRole('button', { name: 'Upload' }).click();

        const fileChooser = await fileChooserPromise;
        await fileChooser.setFiles([{ name, mimeType: 'application/zip', buffer: Buffer.from('fake zip content') }]);
    }

    getPreparingStatus() {
        return this.getDialog().getByText('Preparing');
    }

    getStatisticsHeading() {
        return this.getDialog().getByText('Imported dataset statistics');
    }

    getLabelMappingHeading() {
        return this.getDialog().getByText('Label mapping');
    }

    getLabelMappingButton(label: string) {
        return this.getDialog().getByRole('button', { name: `${label} Target label for ${label}` });
    }

    getIncludeUnannotatedCheckbox() {
        return this.getDialog().getByRole('checkbox', { name: 'include unannotated' });
    }

    async submit() {
        await this.getDialog().getByRole('button', { name: 'Submit' }).click();
    }

    getImportStatusText(filename: string, status: 'processing' | 'success') {
        const text =
            status === 'processing'
                ? `${filename} file is being processed for import`
                : `${filename} file has been imported successfully`;
        return this.page.getByText(text);
    }

    async closeImportStatus() {
        await this.page.getByRole('button', { name: /close import dataset status/i }).click();
    }

    async cancelPrepareJobInDialog() {
        await this.getDialog().getByRole('button', { name: 'Cancel' }).click();
    }

    async cancelJobFromStatusCard() {
        await this.page.getByRole('button', { name: /cancel job dialog/i, exact: true }).click();

        const alertDialog = this.page.getByRole('alertdialog');
        await alertDialog.getByRole('button', { name: /Cancel Job/i, exact: true }).click();
    }

    getProcessingStatusText(filename: string) {
        return this.page.getByText(`${filename} file is being processed for import`);
    }
}
