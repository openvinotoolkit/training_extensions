// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';

export class JobsDialogPage {
    constructor(private page: Page) {}

    async gotoProjectList() {
        await this.page.goto(paths.project.index({}));
    }

    async gotoProject(projectId: string = 'id-1') {
        await this.page.goto(paths.project.models({ projectId }));
    }

    getJobsButton() {
        return this.page.getByRole('button', { name: 'Jobs' });
    }

    getJobsButtonDot() {
        return this.page.getByTestId('running-job-indicator');
    }

    async openDialog() {
        await this.getJobsButton().click();
    }

    getHeading(title: string) {
        return this.page.getByRole('heading', { name: title, exact: true });
    }

    getProjectPicker() {
        return this.page.getByRole('button', { name: /Filter jobs by project/ });
    }

    async selectProject(optionName: string) {
        await this.getProjectPicker().click();
        await this.page.getByRole('option', { name: optionName }).click();
    }

    getTab(label: string) {
        return this.page.getByRole('tab', { name: new RegExp(`^${label}\\b`) });
    }

    async selectTab(label: string) {
        await this.getTab(label).click();
    }

    getJobByName(name: string) {
        return this.page.getByText(name, { exact: true });
    }

    getStatusBadge(label: string) {
        return this.page.getByText(label, { exact: true });
    }

    getActionsMenuButton(jobName: string) {
        return this.page.getByLabel(`Job actions for ${jobName}`);
    }

    async openActionsMenu(jobName: string) {
        await this.getActionsMenuButton(jobName).click();
    }

    getCancelMenuItem() {
        return this.page.getByRole('menuitem', { name: 'Cancel' });
    }

    getViewLogsMenuItem() {
        return this.page.getByRole('menuitem', { name: 'View logs' });
    }

    async cancelJob(jobName: string) {
        await this.openActionsMenu(jobName);
        await this.getCancelMenuItem().click();
        await this.page
            .getByRole('alertdialog', { name: 'Cancel job' })
            .getByRole('button', { name: 'Cancel job' })
            .click();
    }

    async viewLogs(jobName: string) {
        await this.openActionsMenu(jobName);
        await this.getViewLogsMenuItem().click();
    }

    getLogsHeading() {
        return this.page.getByRole('heading', { name: 'Training Logs', exact: true });
    }

    async closeLogsDialog() {
        await this.page.getByRole('button', { name: 'Close dialog' }).click();
    }
}
