// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isNonEmptyString } from '../../../../../shared/util';

type LabelsMapping = Record<string, string | null>;

export const IMPORT_DATASET_FORM_ID = 'import-dataset-form';
export const UNMAPPED_LABEL_VALUE = '__UNMAPPED__';

const isValidLabel = (label: unknown): label is string => {
    return isNonEmptyString(label) && label !== UNMAPPED_LABEL_VALUE;
};

export const mapProjectLabels = (datasetLabels: string[], formData: FormData): LabelsMapping => {
    return datasetLabels.reduce<LabelsMapping>((acc, sourceLabel, index) => {
        const targetLabel = formData.get(`targetLabel-${index}`);

        acc[sourceLabel] = isValidLabel(targetLabel) ? targetLabel : null;

        return acc;
    }, {});
};
