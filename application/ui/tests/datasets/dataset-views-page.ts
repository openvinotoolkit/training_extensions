// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Page } from '@playwright/test';

export class DatasetViewsPage {
    constructor(private readonly page: Page) {}

    getViewSelectorTrigger() {
        return this.page.getByRole('button', { name: 'Select dataset view' });
    }

    async openViewSelector() {
        await this.getViewSelectorTrigger().click();
    }

    getViewsList() {
        return this.page.getByRole('list', { name: 'Dataset views list' });
    }

    getViewRow(name: string) {
        return this.getViewsList().getByRole('listitem', { name });
    }

    async selectView(name: string) {
        await this.openViewSelector();
        await this.getViewRow(name).click();
    }

    async openViewActions(name: string) {
        await this.page.getByRole('button', { name: `Dataset view actions for ${name}` }).click();
    }

    async renameView(name: string, newName: string) {
        await this.openViewSelector();
        await this.openViewActions(name);
        await this.page.getByRole('menuitem', { name: 'Rename' }).click();

        const dialog = this.page.getByRole('dialog');
        await dialog.getByLabel('View name').fill(newName);
        await dialog.getByRole('button', { name: 'Save' }).click();
    }

    async deleteView(name: string, confirm = true) {
        await this.openViewSelector();
        await this.openViewActions(name);
        await this.page.getByRole('menuitem', { name: 'Delete' }).click();

        const dialog = this.page.getByRole('alertdialog');
        await dialog.getByRole('button', { name: confirm ? 'Delete' : 'Close' }).click();
    }

    getSaveViewButton() {
        return this.page.getByRole('button', { name: 'Save view' });
    }

    async saveViewAs(name: string) {
        await this.getSaveViewButton().click();

        const dialog = this.page.getByRole('dialog');
        await dialog.getByLabel('View name').fill(name);
        await dialog.getByRole('button', { name: 'Save' }).click();
    }

    getAssignButton() {
        return this.page.getByRole('button', { name: 'Assign to existing view' });
    }

    async assignToView(name: string) {
        await this.getAssignButton().click();

        const dialog = this.page.getByRole('dialog');
        await dialog.getByRole('button', { name: /Assign to/i }).click();
        await this.page.getByRole('option', { name, exact: true }).click();
        await dialog.getByRole('button', { name: 'Assign', exact: true }).click();
    }

    getUnassignButton() {
        return this.page.getByRole('button', { name: 'Unassign from this view' });
    }

    getEmptyViewMessage() {
        return this.page.getByText('This view has no media items.');
    }

    getGoToEntireDatasetButton() {
        return this.page.getByRole('button', { name: 'Go to Entire dataset' });
    }

    getOpenViewToastLink(name: string) {
        return this.page.getByLabel('toast').getByRole('link', { name: `Open ${name} view` });
    }

    getDeletedSuccessToast(name: string) {
        return this.page.getByText(`Dataset view "${name}" has been deleted successfully.`);
    }
}
