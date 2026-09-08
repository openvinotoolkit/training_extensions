// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { paths } from '../../src/constants/paths';

export class ModelsPage {
    constructor(private page: Page) {}

    async goto(projectId: string = 'id-1') {
        await this.page.goto(paths.project.models({ projectId }));
    }

    getGroupByPicker() {
        return this.page.getByRole('button', { name: 'Group models' });
    }

    async selectGroupBy(option: 'dataset' | 'architecture') {
        await this.getGroupByPicker().click();
        await this.page.getByRole('option', { name: option }).click();
    }

    getModelGroup(groupId: string) {
        return this.page.getByTestId(`model-group-${groupId}`);
    }

    getColumnHeader(groupId: string, label: string) {
        // The accessible name changes once a column becomes the one being sorted on.
        return this.getModelGroup(groupId).getByRole('button', {
            name: new RegExp(`^(Sort by ${label}$|${label}, sorted )`),
        });
    }

    async sortByColumn(groupId: string, label: string) {
        await this.getColumnHeader(groupId, label).click();
    }

    async selectPickerOption(label: string, optionName: string) {
        await this.page.getByLabel(label, { exact: true }).last().click();
        await this.page.getByRole('option', { name: optionName, exact: true }).click();
    }

    async openTrainModelDialog() {
        await this.page.getByRole('button', { name: 'Train model' }).click();
    }

    getModelArchitecture(architectureName: string) {
        return this.page.getByRole('radio', { name: architectureName, exact: true });
    }

    async selectModelArchitecture(architectureName: string) {
        await this.getModelArchitecture(architectureName).click();
    }

    async startTraining() {
        await this.page.getByRole('button', { name: 'Start' }).click();
    }

    async openModelListingOptionsMenu() {
        await this.page.getByRole('button', { name: 'Model listing options' }).click();
    }

    async toggleShowHideFailedModels() {
        await this.openModelListingOptionsMenu();
        await this.page.getByRole('menuitem', { name: /Show failed models|Hide failed models/ }).click();
    }

    getSearchInput() {
        return this.page.getByLabel('Search models');
    }

    async expandSearch() {
        await this.page.getByRole('button', { name: 'Search models' }).click();
    }

    async searchModels(query: string) {
        await this.expandSearch();
        await this.getSearchInput().fill(query);
    }

    getModelRows(groupId?: string) {
        const root = groupId === undefined ? this.page : this.getModelGroup(groupId);

        return root.getByTestId(/^model-disclosure-/);
    }

    async getModelName() {
        return this.page.getByTestId('model-name').textContent();
    }

    getModelByName(name: string) {
        return this.page.getByTestId('model-name').filter({ hasText: name });
    }

    getModelDisclosure(modelId: string) {
        return this.page.getByTestId(`model-disclosure-${modelId}`);
    }

    async expandModel(name: string) {
        await this.getModelByName(name).click();
    }

    async openModelMenu() {
        await this.page.getByLabel('Model actions').first().click();
    }

    async clickRenameAction() {
        await this.page.getByRole('menuitem', { name: 'Rename' }).click();
    }

    async clickDeleteAction() {
        await this.page.getByRole('menuitem', { name: 'Delete model' }).click();
    }

    async clickDeleteWeightsAction() {
        await this.page.getByRole('menuitem', { name: 'Delete weights' }).click();
    }

    async renameModel(newName: string) {
        const textbox = this.page.getByRole('textbox', { name: 'Model name' });

        await textbox.fill(newName);
        await textbox.press('Enter');
    }

    async confirmDeleteModel() {
        await this.page.getByRole('button', { name: 'Delete model', exact: true }).click();
    }

    async confirmDeleteWeights() {
        await this.page.getByRole('button', { name: 'Delete weights' }).click();
    }

    async confirmDeleteDataset() {
        await this.page.getByRole('button', { name: 'Delete', exact: true }).click();
    }

    async getModelNamesInOrder(groupId?: string) {
        return this.getModelRows(groupId).getByTestId('model-name').allTextContents();
    }

    async openDatasetMenu() {
        await this.page.getByLabel('Dataset actions').first().click();
    }

    async clickRenameDatasetAction() {
        await this.page.getByRole('menuitem', { name: 'Rename' }).click();
    }

    async clickDeleteDatasetAction() {
        await this.page.getByRole('menuitem', { name: 'Delete' }).click();
    }

    async renameDatasetRevision(newName: string) {
        const textbox = this.page.getByRole('textbox', { name: 'Dataset revision name' });

        await textbox.fill(newName);
        await textbox.press('Enter');
    }

    getDatasetHeaderByName(name: string) {
        return this.page.getByRole('heading').filter({ hasText: name });
    }

    getThreeSectionRange(datasetId: string) {
        return this.page.getByTestId(`dataset-range-${datasetId}`);
    }

    async clickTrainingDatasetsTab() {
        await this.page.getByRole('tab', { name: 'Training datasets' }).click();
    }

    async clickModelVariantsTab() {
        await this.page.getByRole('tab', { name: 'Model variants' }).click();
    }

    async openAdvancedSettings() {
        await this.page.getByRole('button', { name: 'Advanced settings' }).click();
    }

    async openMoreModelArchitectures() {
        await this.page.getByRole('button', { name: 'Show more' }).click();
    }

    async goBack() {
        await this.page.getByRole('button', { name: 'Back' }).click();
    }

    async openTrainingParameters() {
        await this.page.getByRole('tab', { name: 'Training' }).click();
    }

    async updateInputSizeParameters(inputSizeWidth: number, inputSizeHeight: number) {
        await this.page.getByRole('button', { name: 'Select Input size width' }).click();
        await this.page
            .getByRole('listbox', { name: 'Select Input size width' })
            .getByRole('option', { name: inputSizeWidth.toString() })
            .click();

        await this.page.getByRole('button', { name: 'Select Input size height' }).click();
        await this.page
            .getByRole('listbox', { name: 'Select Input size height' })
            .getByRole('option', { name: inputSizeHeight.toString() })
            .click();
    }

    getQuantizationDialog() {
        return this.page.getByRole('dialog');
    }

    async openQuantizationDialog() {
        await this.page.getByRole('button', { name: 'Start quantization' }).click();
    }

    getAccuracyDropInput() {
        return this.getQuantizationDialog().getByRole('textbox', { name: 'Change Max accuracy drop' });
    }

    getCalibrationSizeInput() {
        return this.getQuantizationDialog().getByRole('textbox', { name: 'Change Max calibration size' });
    }

    getMaxNumIterationsInput() {
        return this.getQuantizationDialog().getByRole('textbox', { name: 'Change Max number of iterations' });
    }

    getNoMaximumCheckbox() {
        return this.getQuantizationDialog().getByLabel('No maximum');
    }

    async submitQuantization() {
        await this.getQuantizationDialog().getByRole('button', { name: 'Start quantization' }).click();
    }

    getToast(message: string) {
        return this.page.getByLabel('toast').filter({ hasText: message });
    }

    getRecommendedModelArchitectures() {
        return this.page.getByLabel('Recommended model architectures');
    }

    getRunningJob(modelName: string) {
        return this.page.getByLabel('Current jobs').getByText(modelName).first();
    }

    getModelVariantRow(modelName: string, propertyName: string) {
        return this.page
            .getByRole('group', { name: modelName })
            .getByLabel(/Model variants for/)
            .getByRole('row', { name: propertyName });
    }

    getModelVariantAccuracy(modelName: string, propertyName: string, precision: string) {
        return this.getModelVariantRow(modelName, propertyName)
            .getByTestId(`model-variant-value-accuracy-${precision.toLocaleLowerCase()}`)
            .getAttribute('data-value');
    }
}
