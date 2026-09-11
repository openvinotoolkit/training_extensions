// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TrainingConfigurationParameter } from '@/api/types';
import { createI18nInstance } from '@/i18n';
import { getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';

import { findGroupByKey, flattenParameters, isParameterGroup } from './utils';

describe('Training parameters utils', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    it('finds a nested top-level group by key', () => {
        const parameters = getMockedTrainingConfiguration();

        const group = findGroupByKey(parameters, 'dataset_preparation');

        expect(group?.name).toBe('Dataset preparation');
    });

    it('returns undefined when group key does not exist', () => {
        const parameters = getMockedTrainingConfiguration();

        expect(findGroupByKey(parameters, 'non_existing_key')).toBeUndefined();
    });

    it('identifies parameter groups correctly', () => {
        const parameters = getMockedTrainingConfiguration();
        const trainingGroup = findGroupByKey(parameters, 'training');

        expect(isParameterGroup(parameters[0])).toBe(true);
        expect(isParameterGroup(trainingGroup?.parameters[0] as TrainingConfigurationParameter)).toBe(false);
    });

    it('flattens nested groups and formats values', () => {
        const parameters = getMockedTrainingConfiguration();
        const datasetPreparationGroup = findGroupByKey(parameters, 'dataset_preparation');
        const augmentationGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'augmentation');

        const rows = flattenParameters(augmentationGroup?.parameters, t);

        expect(rows).toEqual([
            { name: 'Mosaic:', value: '', depth: 0, isGroup: true },
            { name: 'Enable', value: 'On', depth: 1, isGroup: false },
            { name: 'Gaussian blur:', value: '', depth: 0, isGroup: true },
            { name: 'Sigma range', value: '0.1 - 2', depth: 1, isGroup: false },
            { name: 'Probability', value: '0.5', depth: 1, isGroup: false },
        ]);
    });

    it('returns an empty list when parameters are missing', () => {
        expect(flattenParameters(undefined, t)).toEqual([]);
    });
});
