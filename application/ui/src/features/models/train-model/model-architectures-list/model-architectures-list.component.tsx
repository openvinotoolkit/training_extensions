// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Flex } from '@geti-ui/ui';

import { useTrainModelState } from '../train-model-provider.component';
import { AllModelArchitectures } from './all-model-architectures.component';
import { RecommendedModelArchitectures } from './recommended-model-architectures.component';
import { getRecommendedArchitectures } from './utils';

const SHOW_MORE_THRESHOLD = 4;

export const ModelArchitecturesList = () => {
    const { t } = useTranslation();
    const {
        modelArchitectures,
        selectedModelArchitectureId,
        onSelectModelArchitectureId,
        showMoreModelArchitectures,
        onToggleShowMoreModelArchitectures,
    } = useTrainModelState();

    const recommendedArchitectures = getRecommendedArchitectures(modelArchitectures);
    const collapsedArchitectures = recommendedArchitectures.slice(0, SHOW_MORE_THRESHOLD);
    const canToggleArchitecturesList = modelArchitectures.length > SHOW_MORE_THRESHOLD;

    return (
        <Flex direction={'column'} minHeight={0} gap={'size-300'}>
            {showMoreModelArchitectures ? (
                <AllModelArchitectures
                    modelArchitectures={modelArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
                />
            ) : (
                <RecommendedModelArchitectures
                    modelArchitectures={collapsedArchitectures}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectModelArchitectureId}
                />
            )}

            {canToggleArchitecturesList && (
                <Button
                    alignSelf={'start'}
                    variant={'primary'}
                    onPress={() => onToggleShowMoreModelArchitectures(!showMoreModelArchitectures)}
                >
                    {showMoreModelArchitectures
                        ? t('models.training.architectures.list.showLess')
                        : t('models.training.architectures.list.showMore')}
                </Button>
            )}
        </Flex>
    );
};
