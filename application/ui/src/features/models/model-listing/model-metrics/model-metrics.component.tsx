// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Evaluation } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Loading, Text } from '@geti-ui/ui';

import { useGetModelTrainingMetrics } from '../../hooks/api/use-get-model-training-metrics.hook';
import { ModelEvaluations } from './model-evaluations.component';
import { ModelMetricsGraphs } from './model-metrics-graphs.component';

type ModelMetricsProps = {
    modelId: string;
    evaluations: Evaluation[];
    filesDeleted?: boolean;
};

export const ModelMetrics = ({ modelId, evaluations, filesDeleted = false }: ModelMetricsProps) => {
    const { t } = useTranslation();
    const { data: trainingMetrics, isPending, isError } = useGetModelTrainingMetrics(filesDeleted ? null : modelId);

    if (filesDeleted) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                <Text>{t('models.metrics.noMetrics')}</Text>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap={'size-300'}>
            {isPending ? (
                <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                    <Loading size={'M'} mode={'inline'} />
                </Flex>
            ) : isError ? (
                <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                    <Text>{t('models.metrics.loadError')}</Text>
                </Flex>
            ) : (
                <>
                    <ModelEvaluations evaluations={evaluations} />
                    <ModelMetricsGraphs trainingMetrics={trainingMetrics?.training_metrics ?? []} />
                </>
            )}
        </Flex>
    );
};
