// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TFunction } from 'i18next';

export const validateProjectName = (name: string, projectNames: string[], t: TFunction): string | undefined => {
    if (name.trim().length === 0) {
        return t('project.validation.nameEmpty');
    }

    if (projectNames.includes(name)) {
        return t('project.validation.nameExists');
    }

    return undefined;
};

export const PROJECT_NAME_MAX_LENGTH = 100;
