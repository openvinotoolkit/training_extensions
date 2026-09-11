// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ConfigurableParameter, ConfigurableParameterGroup, TrainingConfiguration } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { findGroupByKey, isParameter } from '../../../../model-listing/model-training-parameters/utils';
import { isBoolParameter } from '../../utils';

export type TilingConfigurableParameterGroup = Omit<ConfigurableParameterGroup, 'parameters'> & {
    parameters: ConfigurableParameter[];
};

export const getTilingParameters = (
    trainingConfiguration: TrainingConfiguration
): TilingConfigurableParameterGroup | undefined => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;
    const dataAugmentation = findGroupByKey(datasetPreparation, 'augmentation')?.parameters;
    const tilingParameters = findGroupByKey(dataAugmentation, 'tiling');

    if (tilingParameters === undefined || tilingParameters.parameters === undefined) return undefined;

    return {
        ...tilingParameters,
        parameters: tilingParameters.parameters.filter(isParameter),
    };
};

export const getTilingOffDescription = (t: TranslateFn): string =>
    t('models.training.dataManagement.tiling.offDescription');

export const getTilingAutomaticDescription = (t: TranslateFn): string =>
    t('models.training.dataManagement.tiling.automaticDescription');

const ADAPTIVE_TILING_PARAMETER = 'enable_adaptive_tiling';
const ENABLE_TILING_PARAMETER = 'enable';

const getBoolParameter = (tilingParameters: ConfigurableParameter[], key: string) => {
    const parameter = tilingParameters.find((tilingParameter) => key === tilingParameter.key);

    if (parameter === undefined || !isBoolParameter(parameter)) {
        return undefined;
    }

    return parameter;
};

export const getAdaptiveTilingParameter = (tilingParameters: ConfigurableParameter[]) => {
    return getBoolParameter(tilingParameters, ADAPTIVE_TILING_PARAMETER);
};

export const getEnableTilingParameter = (tilingParameters: ConfigurableParameter[]) => {
    return getBoolParameter(tilingParameters, ENABLE_TILING_PARAMETER);
};

export const TILING_MODES = {
    OFF: 'Off',
    AUTOMATIC: 'Automatic',
    CUSTOM: 'Custom',
} as const;

export type TilingMode = (typeof TILING_MODES)[keyof typeof TILING_MODES];

export const getTilingMode = (tilingParameters: ConfigurableParameter[]): TilingMode => {
    const adaptive = getAdaptiveTilingParameter(tilingParameters);
    const enablingTiling = getEnableTilingParameter(tilingParameters);

    if (!enablingTiling || enablingTiling.value === false) {
        return TILING_MODES.OFF;
    }

    if (adaptive?.value === true) {
        return TILING_MODES.AUTOMATIC;
    }

    return TILING_MODES.CUSTOM;
};

export const getTilingModeLabel = (mode: TilingMode, t: TranslateFn): string => {
    switch (mode) {
        case TILING_MODES.OFF:
            return t('models.training.dataManagement.tiling.modes.off');
        case TILING_MODES.AUTOMATIC:
            return t('models.training.dataManagement.tiling.modes.automatic');
        case TILING_MODES.CUSTOM:
            return t('models.training.dataManagement.tiling.modes.custom');
    }
};

export const getCustomTilingParameters = (parameters: ConfigurableParameter[]) => {
    return parameters.filter(
        (parameter) => ![ADAPTIVE_TILING_PARAMETER, ENABLE_TILING_PARAMETER].includes(parameter.key)
    );
};
